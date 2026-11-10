import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from batch_test_inference import (
    build_network,
    compute_metrics,
    make_paths,
    prepare_condition,
    read_ascii_raster,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_indices(args):
    if args.indices:
        indices = []
        for item in args.indices.split(","):
            item = item.strip()
            if not item:
                continue
            indices.append(int(item))
        return sorted(set(indices))

    if args.start is None or args.end is None:
        raise ValueError("Use either --indices or both --start and --end.")

    return list(range(args.start, args.end + 1))


def load_ground_truth(gt_path, depth_threshold):
    gt_depth = read_ascii_raster(gt_path)
    gt_depth[gt_depth < depth_threshold] = 0.0
    return gt_depth


def denormalize_prediction(pred_norm, depth_sqrt_scale, depth_threshold):
    """Convert the normalized square-root target to depth without filtering."""
    pred_sqrt = (pred_norm + 1.0) / 2.0 * depth_sqrt_scale
    pred_depth = pred_sqrt ** 2
    pred_depth[pred_depth < depth_threshold] = 0.0
    return pred_depth


def run_one_sample(net, cond_tensor, device, args, sample_idx, member_idx):
    set_seed(args.seed + sample_idx * 10000 + member_idx)
    with torch.no_grad():
        pred_noise, _ = net.restoration(cond_tensor, sample_num=args.sample_num)

    pred_norm = pred_noise.squeeze().cpu().numpy()
    return denormalize_prediction(
        pred_norm,
        depth_sqrt_scale=args.depth_sqrt_scale,
        depth_threshold=args.depth_threshold,
    )


def summarize_member_metrics(member_rows, sample_idx):
    keys = [
        "rmse_m",
        "mae_m",
        "csi",
        "f1",
        "iou",
        "flood_area_error_ratio",
        "max_depth_error_m",
        "abs_max_depth_error_m",
        "pred_wet_fraction",
        "pred_max_depth_m",
    ]
    rows = [row for row in member_rows if row["sample_idx"] == sample_idx and row["status"] == "ok"]
    summary = {}
    for key in keys:
        values = np.array([float(row[key]) for row in rows if row.get(key) not in ("", None)], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        summary[f"member_{key}_mean"] = float(np.mean(values))
        summary[f"member_{key}_std"] = float(np.std(values))
        summary[f"member_{key}_min"] = float(np.min(values))
        summary[f"member_{key}_max"] = float(np.max(values))
    return summary


def save_uq_arrays(output_dir, base_name, mean_depth, median_depth, variance_depth, wet_prob):
    np.save(os.path.join(output_dir, f"ensemble_mean_{base_name}.npy"), mean_depth)
    np.save(os.path.join(output_dir, f"ensemble_median_{base_name}.npy"), median_depth)
    np.save(os.path.join(output_dir, f"ensemble_variance_{base_name}.npy"), variance_depth)
    np.save(os.path.join(output_dir, f"wet_probability_{base_name}.npy"), wet_prob)


def save_members(output_dir, base_name, samples):
    member_dir = os.path.join(output_dir, "members", base_name)
    os.makedirs(member_dir, exist_ok=True)
    for idx, sample in enumerate(samples):
        np.save(os.path.join(member_dir, f"member_{idx:03d}.npy"), sample)


def save_uq_figure(output_path, dem_data, rain_data, gt_depth, mean_depth, median_depth, variance_depth, wet_prob, metrics):
    cmap_water = plt.get_cmap("jet").copy()
    cmap_water.set_bad(color="white")

    masked_mean = np.ma.masked_where(mean_depth <= 0, mean_depth)
    masked_median = np.ma.masked_where(median_depth <= 0, median_depth)
    masked_gt = np.ma.masked_where(gt_depth <= 0, gt_depth)
    error_map = mean_depth - gt_depth

    depth_vmax = max(float(mean_depth.max()), float(median_depth.max()), float(gt_depth.max()), 0.1)
    err_max = float(np.max(np.abs(error_map)))
    err_max = err_max if err_max > 0.01 else 0.1
    var_vmax = float(np.percentile(variance_depth, 99))
    var_vmax = var_vmax if var_vmax > 1e-6 else 1e-6

    plt.figure(figsize=(30, 10))

    plt.subplot(2, 4, 1)
    plt.title("Input DEM")
    plt.imshow(dem_data, cmap="terrain", interpolation="nearest")
    plt.colorbar(label="Elevation (m)")

    plt.subplot(2, 4, 2)
    plt.title("Input Cumulative Rain")
    plt.imshow(rain_data, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Rainfall (mm)")

    plt.subplot(2, 4, 3)
    plt.title(f"GT Depth (Max: {gt_depth.max():.2f}m)")
    plt.imshow(masked_gt, cmap=cmap_water, vmin=0, vmax=depth_vmax, interpolation="nearest")
    plt.colorbar(label="Water Depth (m)")

    plt.subplot(2, 4, 4)
    plt.title(f"Ensemble Mean (RMSE: {metrics['mean_rmse_m']:.3f}m)")
    plt.imshow(masked_mean, cmap=cmap_water, vmin=0, vmax=depth_vmax, interpolation="nearest")
    plt.colorbar(label="Water Depth (m)")

    plt.subplot(2, 4, 5)
    plt.title("Ensemble Median")
    plt.imshow(masked_median, cmap=cmap_water, vmin=0, vmax=depth_vmax, interpolation="nearest")
    plt.colorbar(label="Water Depth (m)")

    plt.subplot(2, 4, 6)
    plt.title("Wet Probability")
    plt.imshow(wet_prob, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(label="P(wet)")

    plt.subplot(2, 4, 7)
    plt.title("Depth Variance")
    plt.imshow(variance_depth, cmap="magma", vmin=0, vmax=var_vmax, interpolation="nearest")
    plt.colorbar(label="Variance (m2)")

    plt.subplot(2, 4, 8)
    plt.title("Mean Error [Pred - GT]")
    plt.imshow(error_map, cmap="RdBu_r", vmin=-err_max, vmax=err_max, interpolation="nearest")
    plt.colorbar(label="Error (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="UQ ensemble sampling for selected flood cases.")
    parser.add_argument("--start", type=int, default=None, help="Inclusive sample index.")
    parser.add_argument("--end", type=int, default=None, help="Inclusive sample index.")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated sample indices, e.g. 1,4,6,9.")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of diffusion samples per condition.")
    parser.add_argument("--data-root", type=str, default="datasets/flood")
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/wd_cddpm_ema.pth",
    )
    parser.add_argument("--output-dir", type=str, default="uq_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--sample-num", type=int, default=1)
    parser.add_argument(
        "--max-rain",
        type=float,
        default=100.0,
        help="Normalization scale for each ten-minute rainfall channel (mm).",
    )
    parser.add_argument("--depth-sqrt-scale", type=float, default=3.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    parser.add_argument("--cell-size", type=float, default=8.35)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-members", action="store_true", help="Save every ensemble member depth map.")
    parser.add_argument("--no-save-figs", action="store_true", help="Skip PNG figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    indices = parse_indices(args)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    arrays_dir = os.path.join(args.output_dir, "arrays")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(arrays_dir, exist_ok=True)
    if not args.no_save_figs:
        os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading model from: {args.weights}")
    net = build_network(test_steps=args.test_steps)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)
    net.set_new_noise_schedule(device=device, phase="test")
    net.eval()

    member_rows = []
    ensemble_rows = []

    for sample_idx in tqdm(indices, desc="uq cases"):
        paths = make_paths(args.data_root, sample_idx)
        base_name = f"dem_{paths['dem_id']}_rain_{paths['rain_id']}"

        missing = [key for key in ("dem", "rain", "gt") if not os.path.exists(paths[key])]
        if missing:
            message = ";".join(f"{key}:{paths[key]}" for key in missing)
            ensemble_rows.append({
                "sample_idx": sample_idx,
                "dem_id": paths["dem_id"],
                "rain_id": paths["rain_id"],
                "res_id": paths["res_id"],
                "status": "missing_input",
                "message": message,
            })
            continue

        try:
            cond_input, dem_data, rain_data = prepare_condition(paths, args.max_rain)
            cond_tensor = torch.FloatTensor(cond_input).unsqueeze(0).to(device)
            gt_depth = load_ground_truth(paths["gt"], args.depth_threshold)

            samples = []
            for member_idx in tqdm(range(args.n_samples), desc=base_name, leave=False):
                pred_depth = run_one_sample(net, cond_tensor, device, args, sample_idx, member_idx)
                samples.append(pred_depth)

                metrics = compute_metrics(pred_depth, gt_depth, cell_size=args.cell_size)
                pred_wet_fraction = float((pred_depth > 0).sum() / pred_depth.size)
                member_rows.append({
                    "sample_idx": sample_idx,
                    "dem_id": paths["dem_id"],
                    "rain_id": paths["rain_id"],
                    "res_id": paths["res_id"],
                    "member_idx": member_idx,
                    "status": "ok",
                    "message": "",
                    "pred_wet_fraction": pred_wet_fraction,
                    **metrics,
                })

            sample_stack = np.stack(samples, axis=0)
            mean_depth = sample_stack.mean(axis=0)
            mean_depth[mean_depth < args.depth_threshold] = 0.0
            median_depth = np.median(sample_stack, axis=0)
            median_depth[median_depth < args.depth_threshold] = 0.0
            variance_depth = sample_stack.var(axis=0)
            wet_prob = (sample_stack >= args.depth_threshold).mean(axis=0)

            mean_metrics = compute_metrics(mean_depth, gt_depth, cell_size=args.cell_size)
            median_metrics = compute_metrics(median_depth, gt_depth, cell_size=args.cell_size)

            ensemble_row = {
                "sample_idx": sample_idx,
                "dem_id": paths["dem_id"],
                "rain_id": paths["rain_id"],
                "res_id": paths["res_id"],
                "status": "ok",
                "message": "",
                "n_samples": args.n_samples,
                "mean_wet_fraction": float((mean_depth > 0).sum() / mean_depth.size),
                "median_wet_fraction": float((median_depth > 0).sum() / median_depth.size),
                "mean_variance_depth_m2": float(variance_depth.mean()),
                "p95_variance_depth_m2": float(np.percentile(variance_depth, 95)),
            }
            ensemble_row.update({f"mean_{key}": value for key, value in mean_metrics.items()})
            ensemble_row.update({f"median_{key}": value for key, value in median_metrics.items()})
            ensemble_row.update(summarize_member_metrics(member_rows, sample_idx))
            ensemble_rows.append(ensemble_row)

            save_uq_arrays(arrays_dir, base_name, mean_depth, median_depth, variance_depth, wet_prob)
            if args.save_members:
                save_members(arrays_dir, base_name, sample_stack)

            if not args.no_save_figs:
                save_uq_figure(
                    os.path.join(figures_dir, f"uq_result_{base_name}.png"),
                    dem_data=dem_data,
                    rain_data=rain_data,
                    gt_depth=gt_depth,
                    mean_depth=mean_depth,
                    median_depth=median_depth,
                    variance_depth=variance_depth,
                    wet_prob=wet_prob,
                    metrics={"mean_rmse_m": mean_metrics["rmse_m"]},
                )

        except Exception as exc:
            ensemble_rows.append({
                "sample_idx": sample_idx,
                "dem_id": paths["dem_id"],
                "rain_id": paths["rain_id"],
                "res_id": paths["res_id"],
                "status": "failed",
                "message": repr(exc),
            })

    index_label = args.indices.replace(",", "_") if args.indices else f"{args.start}_{args.end}"
    member_csv = os.path.join(args.output_dir, f"uq_members_{index_label}.csv")
    ensemble_csv = os.path.join(args.output_dir, f"uq_ensemble_{index_label}.csv")

    member_fieldnames = [
        "sample_idx",
        "dem_id",
        "rain_id",
        "res_id",
        "member_idx",
        "status",
        "message",
        "pred_wet_fraction",
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
        "pred_max_depth_m",
        "gt_max_depth_m",
        "max_depth_error_m",
        "abs_max_depth_error_m",
    ]
    ensemble_fieldnames = sorted({key for row in ensemble_rows for key in row.keys()})

    write_csv(member_csv, member_rows, member_fieldnames)
    write_csv(ensemble_csv, ensemble_rows, ensemble_fieldnames)

    ok_count = sum(row.get("status") == "ok" for row in ensemble_rows)
    print(f"Finished UQ cases. OK: {ok_count}/{len(indices)}")
    print(f"Member metrics CSV: {member_csv}")
    print(f"Ensemble metrics CSV: {ensemble_csv}")
    print(f"Arrays dir: {arrays_dir}")
    if not args.no_save_figs:
        print(f"Figures dir: {figures_dir}")


if __name__ == "__main__":
    main()
