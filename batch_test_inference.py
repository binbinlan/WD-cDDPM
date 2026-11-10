import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

from models.network import Network


def read_ascii_raster(filepath):
    data = np.loadtxt(filepath, skiprows=6)
    data = np.where(data == -9999.0, 0.0, data)
    return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_network(test_steps):
    return Network(
        init_type="kaiming",
        module_name="guided_diffusion",
        unet={
            "in_channel": 15,
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
                "n_timestep": test_steps,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        },
    )


def make_paths(data_root, sample_idx):
    dem_id = f"{sample_idx:02d}"
    rain_id = sample_idx
    res_id = sample_idx + 1
    return {
        "sample_idx": sample_idx,
        "dem_id": dem_id,
        "rain_id": rain_id,
        "res_id": res_id,
        "dem": os.path.join(data_root, "terrain", f"dem_{dem_id}.dem"),
        "rain": os.path.join(
            data_root,
            "cumulative_rainfall_npy",
            "time_slice",
            f"cumulative_rain_seed_{rain_id}.npy",
        ),
        "gt": os.path.join(data_root, "results", f"results_{res_id}", f"seed_{res_id}.max"),
    }


def prepare_condition(paths, max_rain):
    dem_data = read_ascii_raster(paths["dem"])
    rain_data = np.load(paths["rain"], allow_pickle=False)
    if rain_data.ndim != 3 or rain_data.shape[0] != 12:
        raise ValueError(
            f"Rainfall array must have shape [12, H, W], got {rain_data.shape}: "
            f"{paths['rain']}"
        )
    if rain_data.shape[1:] != dem_data.shape:
        raise ValueError(
            f"Rainfall grid {rain_data.shape[1:]} does not match DEM grid "
            f"{dem_data.shape}: {paths['rain']}"
        )

    dy, dx = np.gradient(dem_data)
    slope_data = np.sqrt(dx ** 2 + dy ** 2)

    dem_min, dem_max = dem_data.min(), dem_data.max()
    if dem_max > dem_min:
        dem_norm = (dem_data - dem_min) / (dem_max - dem_min) * 2.0 - 1.0
    else:
        dem_norm = np.zeros_like(dem_data)

    slope_min, slope_max = slope_data.min(), slope_data.max()
    if slope_max > slope_min:
        slope_norm = (slope_data - slope_min) / (slope_max - slope_min) * 2.0 - 1.0
    else:
        slope_norm = np.zeros_like(slope_data)

    rain_norm = np.clip((rain_data / max_rain) * 2.0 - 1.0, -1.0, 1.0)
    cond_input = np.concatenate(
        [dem_norm[None, ...], slope_norm[None, ...], rain_norm], axis=0
    )
    return cond_input, dem_data, rain_data.sum(axis=0, dtype=np.float32)


def denormalize_and_filter(pred_norm, depth_sqrt_scale, depth_threshold, median_size):
    pred_sqrt = (pred_norm + 1.0) / 2.0 * depth_sqrt_scale
    pred_depth = pred_sqrt ** 2
    pred_depth[pred_depth < depth_threshold] = 0.0

    raw_depth = pred_depth.copy()
    if median_size and median_size > 1:
        #pred_depth = ndimage.median_filter(pred_depth, size=median_size)
        pred_depth[pred_depth < depth_threshold] = 0.0

    return raw_depth, pred_depth


def compute_metrics(pred_depth, gt_depth, cell_size):
    error = pred_depth - gt_depth
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    pred_wet = pred_depth > 0
    gt_wet = gt_depth > 0

    tp = int(np.logical_and(pred_wet, gt_wet).sum())
    fp = int(np.logical_and(pred_wet, ~gt_wet).sum())
    fn = int(np.logical_and(~pred_wet, gt_wet).sum())
    tn = int(np.logical_and(~pred_wet, ~gt_wet).sum())

    csi_den = tp + fp + fn
    f1_den = 2 * tp + fp + fn
    csi = float(tp / csi_den) if csi_den else 1.0
    iou = csi
    f1 = float((2 * tp) / f1_den) if f1_den else 1.0

    pred_flood_pixels = int(pred_wet.sum())
    gt_flood_pixels = int(gt_wet.sum())
    flood_area_error_pixels = pred_flood_pixels - gt_flood_pixels
    flood_area_error_m2 = float(flood_area_error_pixels * cell_size * cell_size)
    flood_area_error_ratio = (
        float(flood_area_error_pixels / gt_flood_pixels) if gt_flood_pixels else np.nan
    )
    abs_flood_area_error_m2 = float(abs(flood_area_error_m2))
    abs_flood_area_error_ratio = float(abs(flood_area_error_ratio))

    max_depth_error = float(pred_depth.max() - gt_depth.max())
    abs_max_depth_error = float(abs(max_depth_error))

    return {
        "rmse_m": rmse,
        "mae_m": mae,
        "csi": csi,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_flood_pixels": pred_flood_pixels,
        "gt_flood_pixels": gt_flood_pixels,
        "flood_area_error_pixels": flood_area_error_pixels,
        "flood_area_error_m2": flood_area_error_m2,
        "flood_area_error_ratio": flood_area_error_ratio,
        "abs_flood_area_error_m2": abs_flood_area_error_m2,
        "abs_flood_area_error_ratio": abs_flood_area_error_ratio,
        "pred_max_depth_m": float(pred_depth.max()),
        "gt_max_depth_m": float(gt_depth.max()),
        "max_depth_error_m": max_depth_error,
        "abs_max_depth_error_m": abs_max_depth_error,
    }


def save_figure(output_path, dem_data, rain_data, pred_depth, gt_depth, error_map, metrics):
    cmap_water = plt.get_cmap("jet").copy()
    cmap_water.set_bad(color="white")

    masked_pred = np.ma.masked_where(pred_depth <= 0, pred_depth)
    masked_gt = np.ma.masked_where(gt_depth <= 0, gt_depth)

    pred_max = float(pred_depth.max())
    gt_max = float(gt_depth.max())
    depth_vmax = max(pred_max, gt_max, 0.1)
    max_err = float(np.max(np.abs(error_map)))
    max_err = max_err if max_err > 0.01 else 0.1

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 5, 1)
    plt.title("Input DEM")
    plt.imshow(dem_data, cmap="terrain", interpolation="nearest")
    plt.colorbar(label="Elevation (m)")

    plt.subplot(1, 5, 2)
    plt.title("Input Cumulative Rain")

    plt.imshow(rain_data, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Rainfall (mm)")

    plt.subplot(1, 5, 3)
    plt.title(f"Predicted Depth (Max: {pred_max:.2f}m)")
    plt.imshow(masked_pred, cmap=cmap_water, vmin=0, vmax=depth_vmax, interpolation="nearest")
    plt.colorbar(label="Water Depth (m)")

    plt.subplot(1, 5, 4)
    plt.title(f"Ground Truth (Max: {gt_max:.2f}m)")
    plt.imshow(masked_gt, cmap=cmap_water, vmin=0, vmax=depth_vmax, interpolation="nearest")
    plt.colorbar(label="Water Depth (m)")

    plt.subplot(1, 5, 5)
    plt.title(f"Error Map (RMSE: {metrics['rmse_m']:.3f}m)")
    plt.imshow(error_map, cmap="RdBu_r", vmin=-max_err, vmax=max_err, interpolation="nearest")
    plt.colorbar(label="Error (m) [Pred - GT]")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary_csv(csv_path, rows):
    metric_keys = [
        "rmse_m",
        "mae_m",
        "csi",
        "f1",
        "iou",
        "flood_area_error_m2",
        "flood_area_error_ratio",
        "abs_flood_area_error_m2",
        "abs_flood_area_error_ratio",
        "max_depth_error_m",
        "abs_max_depth_error_m",
    ]
    completed = [row for row in rows if row.get("status") == "ok"]
    if not completed:
        return

    summary_path = os.path.splitext(csv_path)[0] + "_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "median", "std", "min", "max"])
        writer.writeheader()
        for key in metric_keys:
            values = np.array([float(row[key]) for row in completed if row.get(key) not in ("", None)], dtype=float)
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            writer.writerow({
                "metric": key,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })


def backfill_absolute_metrics(csv_path):
    """Add absolute-error columns to a previously generated metrics CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required_sources = {
        "abs_flood_area_error_m2": "flood_area_error_m2",
        "abs_flood_area_error_ratio": "flood_area_error_ratio",
        "abs_max_depth_error_m": "max_depth_error_m",
    }
    missing_sources = [source for source in required_sources.values() if source not in fieldnames]
    if missing_sources:
        raise KeyError(
            f"Cannot backfill {csv_path}; missing source columns: {', '.join(missing_sources)}"
        )

    for output_key, source_key in required_sources.items():
        if output_key not in fieldnames:
            fieldnames.append(output_key)

        for row in rows:
            value = row.get(source_key, "")
            if value in ("", None):
                row[output_key] = ""
                continue
            try:
                row[output_key] = float(abs(float(value)))
            except ValueError:
                row[output_key] = ""

    temp_path = csv_path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, csv_path)

    write_summary_csv(csv_path, rows)
    ok_count = sum(row.get("status") == "ok" for row in rows)
    print(f"Backfilled absolute-error metrics for {ok_count} successful samples: {csv_path}")
    print(f"Updated summary: {os.path.splitext(csv_path)[0]}_summary.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference and evaluation for flood diffusion surrogate.")
    parser.add_argument("--start", type=int, default=5001, help="Inclusive physical sample index, e.g. 1.")
    parser.add_argument("--end", type=int, default=5660,help="Inclusive physical sample index, e.g. 500.")
    parser.add_argument("--data-root", type=str, default="datasets/flood")
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/wd_cddpm_ema.pth",
    )
    parser.add_argument("--output-dir", type=str, default="batch_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-steps", type=int, default=200)
    parser.add_argument("--sample-num", type=int, default=1)
    parser.add_argument(
        "--max-rain",
        type=float,
        default=100.0,
        help="Normalization scale for each ten-minute rainfall channel (mm).",
    )
    parser.add_argument("--depth-sqrt-scale", type=float, default=3.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--cell-size", type=float, default=8.35)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-save-figs", action="store_true", help="Only save predictions and CSV metrics.")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw prediction before median filtering.")
    parser.add_argument(
        "--backfill-csv",
        type=str,
        default=None,
        help=(
            "Update an existing per-event metrics CSV with absolute area/depth error fields "
            "and exit without running model inference."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backfill_csv:
        backfill_absolute_metrics(args.backfill_csv)
        return

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, "predictions")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(pred_dir, exist_ok=True)
    if not args.no_save_figs:
        os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading model from: {args.weights}")
    net = build_network(test_steps=args.test_steps)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)
    net.set_new_noise_schedule(device=device, phase="test")
    net.eval()

    csv_path = os.path.join(args.output_dir, f"metrics_{args.start}_{args.end}.csv")
    rows = []

    for sample_idx in tqdm(range(args.start, args.end + 1), desc="batch inference"):
        paths = make_paths(args.data_root, sample_idx)
        base_name = f"dem_{paths['dem_id']}_rain_{paths['rain_id']}"
        row = {
            "sample_idx": sample_idx,
            "dem_id": paths["dem_id"],
            "rain_id": paths["rain_id"],
            "res_id": paths["res_id"],
            "status": "ok",
            "message": "",
        }

        missing = [key for key in ("dem", "rain", "gt") if not os.path.exists(paths[key])]
        if missing:
            row["status"] = "missing_input"
            row["message"] = ";".join(f"{key}:{paths[key]}" for key in missing)
            rows.append(row)
            continue

        try:
            set_seed(args.seed + sample_idx)
            cond_input, dem_data, rain_data = prepare_condition(paths, args.max_rain)
            cond_tensor = torch.FloatTensor(cond_input).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_noise, _ = net.restoration(cond_tensor, sample_num=args.sample_num)

            pred_norm = pred_noise.squeeze().cpu().numpy()
            raw_depth, pred_depth = denormalize_and_filter(
                pred_norm,
                depth_sqrt_scale=args.depth_sqrt_scale,
                depth_threshold=args.depth_threshold,
                median_size=args.median_size,
            )

            gt_depth = read_ascii_raster(paths["gt"])
            gt_depth[gt_depth < args.depth_threshold] = 0.0

            if pred_depth.shape != gt_depth.shape:
                row["status"] = "shape_mismatch"
                row["message"] = f"pred:{pred_depth.shape};gt:{gt_depth.shape}"
                rows.append(row)
                continue

            error_map = pred_depth - gt_depth
            metrics = compute_metrics(pred_depth, gt_depth, cell_size=args.cell_size)
            row.update(metrics)

            np.save(os.path.join(pred_dir, f"pred_{base_name}.npy"), pred_depth)
            if args.save_raw:
                np.save(os.path.join(pred_dir, f"raw_pred_{base_name}.npy"), raw_depth)

            if not args.no_save_figs:
                save_figure(
                    os.path.join(fig_dir, f"result_{base_name}.png"),
                    dem_data=dem_data,
                    rain_data=rain_data,
                    pred_depth=pred_depth,
                    gt_depth=gt_depth,
                    error_map=error_map,
                    metrics=metrics,
                )

        except Exception as exc:
            row["status"] = "failed"
            row["message"] = repr(exc)

        rows.append(row)

    fieldnames = [
        "sample_idx",
        "dem_id",
        "rain_id",
        "res_id",
        "status",
        "message",
        "rmse_m",
        "mae_m",
        "csi",
        "f1",
        "iou",
        "tp",
        "fp",
        "fn",
        "tn",
        "pred_flood_pixels",
        "gt_flood_pixels",
        "flood_area_error_pixels",
        "flood_area_error_m2",
        "flood_area_error_ratio",
        "abs_flood_area_error_m2",
        "abs_flood_area_error_ratio",
        "pred_max_depth_m",
        "gt_max_depth_m",
        "max_depth_error_m",
        "abs_max_depth_error_m",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_summary_csv(csv_path, rows)

    ok_count = sum(row.get("status") == "ok" for row in rows)
    print(f"Finished. OK: {ok_count}/{len(rows)}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Summary CSV: {os.path.splitext(csv_path)[0]}_summary.csv")
    print(f"Prediction dir: {pred_dir}")
    if not args.no_save_figs:
        print(f"Figure dir: {fig_dir}")


if __name__ == "__main__":
    main()
