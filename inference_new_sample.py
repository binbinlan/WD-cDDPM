"""Run the rainfall-conditioned flood diffusion model on one new case.

The checkpoint was trained on 256 x 256 tiles. By default this script runs
those tiles over the complete input grid and averages overlapping predictions,
so a non-256 DEM can be evaluated without silently resizing the physical data.
Use ``--tile-size 0`` only when there is enough memory for full-grid attention.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.network import Network


MODEL_DOWNSAMPLE_FACTOR = 8  # three U-Net downsampling stages


def _read_ascii_raster(filepath: Path) -> np.ndarray:
    """Read an ASCII raster with the six-line LISFLOOD header."""
    data = np.loadtxt(filepath, skiprows=6).astype(np.float32, copy=False)
    return np.where(data == -9999.0, 0.0, data)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _build_network(weights: Path, device: torch.device) -> Network:
    # image_size remains 256 because that is the checkpoint's training setup;
    # convolutional layers accept padded tiles of the same or smaller size.
    net = Network(
        init_type="kaiming",
        module_name="guided_diffusion",
        unet={
            "in_channel": 15,  # DEM + slope + 12 rainfall channels + noisy depth
            "out_channel": 1,
            "inner_channel": 64,
            "channel_mults": [1, 2, 4, 8],
            "attn_res": [16],
            "num_head_channels": 32,
            "res_blocks": 2,
            "dropout": 0.2,
            "image_size": 256,
        },
        beta_schedule={
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 0.01,
            },
            "test": {
                "schedule": "linear",
                "n_timestep": 1000,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        },
    )
    net.load_state_dict(torch.load(str(weights), map_location=device))
    net.to(device)
    net.set_new_noise_schedule(device=device, phase="test")
    net.eval()
    return net


def _normalise_inputs(dem_data: np.ndarray, rain_data: np.ndarray, max_rain: float) -> np.ndarray:
    if rain_data.ndim != 3 or rain_data.shape[0] != 12:
        raise ValueError(f"Rainfall array must have shape [12, H, W], got {rain_data.shape}")
    if rain_data.shape[1:] != dem_data.shape:
        raise ValueError(f"Rainfall grid {rain_data.shape[1:]} does not match DEM {dem_data.shape}")
    if not np.isfinite(rain_data).all() or np.any(rain_data < 0.0):
        raise ValueError("Rainfall array contains NaN, infinity, or negative values")
    if max_rain <= 0:
        raise ValueError("max_rain must be positive")

    dy, dx = np.gradient(dem_data)
    slope_data = np.sqrt(dx * dx + dy * dy)

    def scale_to_model_range(array: np.ndarray) -> np.ndarray:
        minimum = float(array.min())
        maximum = float(array.max())
        if maximum <= minimum:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - minimum) / (maximum - minimum) * 2.0 - 1.0).astype(np.float32)

    dem_norm = scale_to_model_range(dem_data)
    slope_norm = scale_to_model_range(slope_data)
    rain_norm = np.clip((rain_data / max_rain) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)
    return np.concatenate([dem_norm[None, ...], slope_norm[None, ...], rain_norm], axis=0)


def _pad_condition(condition: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    height, width = condition.shape[1:]
    if target_height < height or target_width < width:
        raise ValueError("Padding target must not be smaller than the input")
    return np.pad(
        condition,
        ((0, 0), (0, target_height - height), (0, target_width - width)),
        mode="edge",
    )


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = [0]
    while starts[-1] + tile_size < length:
        starts.append(min(starts[-1] + stride, length - tile_size))
    return sorted(set(starts))


def _predict_one_tile(
    net: Network,
    condition: np.ndarray,
    device: torch.device,
    tile_size: int,
) -> np.ndarray:
    height, width = condition.shape[1:]
    padded_height = max(tile_size, int(np.ceil(height / MODEL_DOWNSAMPLE_FACTOR)) * MODEL_DOWNSAMPLE_FACTOR)
    padded_width = max(tile_size, int(np.ceil(width / MODEL_DOWNSAMPLE_FACTOR)) * MODEL_DOWNSAMPLE_FACTOR)
    padded = _pad_condition(condition, padded_height, padded_width)
    cond_tensor = torch.from_numpy(padded).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_norm, _ = net.restoration(cond_tensor, sample_num=1)
    return pred_norm[0, 0, :height, :width].detach().cpu().numpy()


def _predict_grid(
    net: Network,
    condition: np.ndarray,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    height, width = condition.shape[1:]
    if tile_size == 0:
        # Full-grid mode is useful for experiments, but global attention scales
        # quadratically with the padded spatial area and can exhaust memory.
        return _predict_one_tile(net, condition, device, tile_size=0)
    if tile_size < MODEL_DOWNSAMPLE_FACTOR or tile_size % MODEL_DOWNSAMPLE_FACTOR:
        raise ValueError(f"tile_size must be 0 or a multiple of {MODEL_DOWNSAMPLE_FACTOR}")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < tile_size")

    stride = tile_size - tile_overlap
    y_starts = _tile_starts(height, tile_size, stride)
    x_starts = _tile_starts(width, tile_size, stride)
    prediction_sum = np.zeros((height, width), dtype=np.float64)
    prediction_count = np.zeros((height, width), dtype=np.float64)
    total_tiles = len(y_starts) * len(x_starts)
    completed = 0
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = condition[:, y0:y1, x0:x1]
            prediction = _predict_one_tile(net, tile, device, tile_size=tile_size)
            prediction_sum[y0:y1, x0:x1] += prediction
            prediction_count[y0:y1, x0:x1] += 1.0
            completed += 1
            print(f"Completed tile {completed}/{total_tiles} ({y0}:{y1}, {x0}:{x1})", flush=True)
    return (prediction_sum / np.maximum(prediction_count, 1.0)).astype(np.float32)


def _depth_from_normalised(pred_norm: np.ndarray, depth_sqrt_scale: float, threshold: float) -> np.ndarray:
    pred_norm = np.clip(pred_norm, -1.0, 1.0)
    pred_sqrt = (pred_norm + 1.0) * 0.5 * depth_sqrt_scale
    pred_depth = pred_sqrt * pred_sqrt
    pred_depth[pred_depth < threshold] = 0.0
    return pred_depth.astype(np.float32, copy=False)


def _save_figure(path: Path, dem: np.ndarray, rain: np.ndarray, depth: np.ndarray) -> None:
    cmap_water = plt.get_cmap("Blues").copy()
    cmap_water.set_bad(color="white")
    masked_depth = np.ma.masked_where(depth <= 0, depth)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input DEM")
    plt.imshow(dem, cmap="gist_earth", interpolation="nearest")
    plt.colorbar(label="Elevation (m)")
    plt.subplot(1, 3, 2)
    plt.title("Input event-total rain")
    plt.imshow(rain.sum(axis=0), cmap="jet", interpolation="nearest")
    plt.colorbar(label="Rainfall (mm)")
    plt.subplot(1, 3, 3)
    plt.title(f"Predicted depth (max {float(depth.max()):.2f} m)")
    plt.imshow(masked_depth, cmap=cmap_water, vmin=0, interpolation="nearest")
    plt.colorbar(label="Water depth (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    data_root = root / "datasets" / "flood"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=data_root / "terrain" / "dem_999.dem")
    parser.add_argument(
        "--rain",
        type=Path,
        default=data_root / "cumulative_rainfall_npy" / "time_slice" / "cumulative_rain_seed_998.npy",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "checkpoints" / "wd_cddpm_ema.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "case_study" / "test_results")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels; use 0 for full-grid mode")
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--max-rain", type=float, default=100.0)
    parser.add_argument("--depth-sqrt-scale", type=float, default=3.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--show", action="store_true", help="Open the saved figure after inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dem = _read_ascii_raster(args.dem)
    rain = np.load(args.rain, allow_pickle=False)
    condition = _normalise_inputs(dem, rain, args.max_rain)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEM shape: {dem.shape}; rainfall shape: {rain.shape}; device: {device}", flush=True)
    print(f"Loading checkpoint: {args.weights}", flush=True)
    net = _build_network(args.weights, device)
    pred_norm = _predict_grid(net, condition, device, args.tile_size, args.tile_overlap)
    pred_depth = _depth_from_normalised(pred_norm, args.depth_sqrt_scale, args.depth_threshold)

    base_name = "dem_999_rain_998"
    npy_path = args.output_dir / f"pred_{base_name}.npy"
    png_path = args.output_dir / f"result_{base_name}.png"
    np.save(npy_path, pred_depth, allow_pickle=False)
    _save_figure(png_path, dem, rain, pred_depth)
    print(f"Prediction saved: {npy_path}", flush=True)
    print(f"Figure saved: {png_path}", flush=True)
    print(f"Output shape: {pred_depth.shape}; wet pixels (>={args.depth_threshold} m): {(pred_depth > 0).sum()}", flush=True)
    if args.show:
        import os

        os.startfile(str(png_path))


if __name__ == "__main__":
    main()



#  python inference_new_sample.py --device cuda --tile-size 0

# Set-Location '<path-to-WD-cDDPM>'
# python inference_new_sample.py --device cuda
