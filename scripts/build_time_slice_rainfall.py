"""Aggregate one-minute LISFLOOD rainfall forcing into 12 ten-minute channels.

Each output keeps the cumulative-rainfall filename and stores a float32 array
with shape ``[12, height, width]``.  Every channel is the rainfall depth (mm)
accumulated over one consecutive ten-minute window.
"""

import argparse
import json
import os
import re
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


FILENAME_PATTERN = re.compile(r"rainfall_seed_(\d+)\.nc$")
DEFAULT_DATA_ROOT = Path("datasets/flood")


ConversionResult = namedtuple(
    "ConversionResult",
    [
        "sample_id",
        "status",
        "window_max_mm",
        "event_total_max_mm",
        "max_total_difference_mm",
        "output_bytes",
    ],
)


def parse_sample_id(path):
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected rainfall filename: {path.name}")
    return int(match.group(1))


def discover_inputs(input_dir, start, end):
    indexed_paths = sorted(
        ((parse_sample_id(path), path) for path in input_dir.glob("rainfall_seed_*.nc")),
        key=lambda item: item[0],
    )
    selected = [
        path
        for sample_id, path in indexed_paths
        if sample_id >= start and (end is None or sample_id <= end)
    ]
    if not selected:
        raise FileNotFoundError(
            f"No rainfall_seed_*.nc files found in {input_dir} for the requested range."
        )
    return selected


def load_time_slice_file(path, expected_channels=12):
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.ndim != 3 or array.shape[0] != expected_channels:
        raise ValueError(
            f"Existing output has shape {array.shape}, expected "
            f"[{expected_channels}, height, width]: {path}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"Existing output is {array.dtype}, expected float32: {path}")
    return array


def convert_one(
    input_path_text,
    cumulative_dir_text,
    output_dir_text,
    window_minutes,
    overwrite,
):
    input_path = Path(input_path_text)
    cumulative_dir = Path(cumulative_dir_text)
    output_dir = Path(output_dir_text)
    sample_id = parse_sample_id(input_path)
    output_path = output_dir / f"cumulative_rain_seed_{sample_id}.npy"

    if output_path.exists() and not overwrite:
        existing = load_time_slice_file(output_path)
        return ConversionResult(
            sample_id=sample_id,
            status="skipped",
            window_max_mm=float(np.max(existing)),
            event_total_max_mm=float(np.max(np.sum(existing, axis=0, dtype=np.float32))),
            max_total_difference_mm=0.0,
            output_bytes=output_path.stat().st_size,
        )

    with Dataset(str(input_path), mode="r") as dataset:
        if "rainfall_depth" not in dataset.variables or "time" not in dataset.variables:
            raise ValueError(f"Missing rainfall_depth or time variable: {input_path}")
        rainfall_variable = dataset["rainfall_depth"]
        time_variable = dataset["time"]
        if getattr(rainfall_variable, "units", None) != "mm":
            raise ValueError(f"rainfall_depth units must be mm: {input_path}")
        if getattr(time_variable, "units", None) != "hour":
            raise ValueError(f"time units must be hour: {input_path}")
        rainfall = np.ma.filled(rainfall_variable[:], np.nan).astype(np.float32)
        time = np.asarray(time_variable[:], dtype=np.float64)

    if rainfall.ndim != 3 or rainfall.shape[0] != time.size:
        raise ValueError(
            f"Rainfall must have shape [time, height, width], got {rainfall.shape}: "
            f"{input_path}"
        )
    if not np.isfinite(rainfall).all() or np.any(rainfall < 0.0):
        raise ValueError(f"Rainfall contains invalid values: {input_path}")
    if time.size < 2:
        raise ValueError(f"At least two time coordinates are required: {input_path}")

    time_step_minutes = float(np.median(np.diff(time)) * 60.0)
    if not np.allclose(np.diff(time) * 60.0, time_step_minutes, atol=1.0e-4, rtol=0.0):
        raise ValueError(f"Nonuniform rainfall time coordinates: {input_path}")
    if not np.isclose(time_step_minutes, 1.0, atol=1.0e-3, rtol=0.0):
        raise ValueError(
            f"Expected one-minute rainfall steps, got {time_step_minutes:.6f} min: "
            f"{input_path}"
        )
    if rainfall.shape[0] % window_minutes != 0:
        raise ValueError(
            f"{rainfall.shape[0]} time steps cannot be divided into "
            f"{window_minutes}-minute windows: {input_path}"
        )

    channel_count = rainfall.shape[0] // window_minutes
    if channel_count != 12:
        raise ValueError(
            f"Expected 12 windows from a two-hour event, got {channel_count}: {input_path}"
        )
    time_slices = rainfall.reshape(
        channel_count, window_minutes, rainfall.shape[1], rainfall.shape[2]
    ).sum(axis=1, dtype=np.float32)

    cumulative_path = cumulative_dir / f"cumulative_rain_seed_{sample_id}.npy"
    if not cumulative_path.exists():
        raise FileNotFoundError(f"Reference cumulative rainfall is missing: {cumulative_path}")
    reference_total = np.load(cumulative_path, allow_pickle=False).astype(np.float32, copy=False)
    reconstructed_total = np.sum(time_slices, axis=0, dtype=np.float32)
    if reference_total.shape != reconstructed_total.shape:
        raise ValueError(
            f"Cumulative shape {reference_total.shape} does not match reconstructed shape "
            f"{reconstructed_total.shape}: {input_path}"
        )
    max_difference = float(np.max(np.abs(reconstructed_total - reference_total)))
    if not np.allclose(reconstructed_total, reference_total, atol=1.0e-3, rtol=1.0e-5):
        raise ValueError(
            f"Ten-minute channels do not reproduce the reference cumulative rainfall "
            f"(max difference {max_difference:.6g} mm): {input_path}"
        )

    temporary_path = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, time_slices, allow_pickle=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return ConversionResult(
        sample_id=sample_id,
        status="created",
        window_max_mm=float(np.max(time_slices)),
        event_total_max_mm=float(np.max(reconstructed_total)),
        max_total_difference_mm=max_difference,
        output_bytes=output_path.stat().st_size,
    )


def write_manifest(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "rainfall",
        help="Directory containing rainfall_seed_<id>.nc files.",
    )
    parser.add_argument(
        "--cumulative-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "cumulative_rainfall_npy",
        help="Directory containing reference cumulative_rain_seed_<id>.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "cumulative_rainfall_npy" / "time_slice",
        help="Destination for the 12-channel arrays.",
    )
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.window_minutes != 10:
        raise ValueError("This B-scheme builder requires --window-minutes 10.")
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("Invalid --start/--end range.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    input_paths = discover_inputs(args.input_dir, args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Converting {len(input_paths)} events to {args.output_dir} "
        f"with {args.workers} workers...",
        flush=True,
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                convert_one,
                str(input_path),
                str(args.cumulative_dir),
                str(args.output_dir),
                args.window_minutes,
                args.overwrite,
            ): input_path
            for input_path in input_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            input_path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Conversion failed for {input_path}") from error
            results.append(result)
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(
                    f"  {completed}/{len(futures)} complete; latest sample "
                    f"{result.sample_id}, max 10-min depth {result.window_max_mm:.3f} mm",
                    flush=True,
                )

    results.sort(key=lambda item: item.sample_id)
    manifest = {
        "source_directory": str(args.input_dir.resolve()),
        "reference_cumulative_directory": str(args.cumulative_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "source_units": "mm per one-minute step",
        "output_units": "mm per ten-minute window",
        "output_shape": [12, 256, 256],
        "channel_intervals_minutes": [
            [index * args.window_minutes, (index + 1) * args.window_minutes]
            for index in range(12)
        ],
        "file_count": len(results),
        "created_count": sum(item.status == "created" for item in results),
        "skipped_count": sum(item.status == "skipped" for item in results),
        "first_sample_id": results[0].sample_id,
        "last_sample_id": results[-1].sample_id,
        "global_max_ten_minute_depth_mm": max(item.window_max_mm for item in results),
        "global_max_event_total_depth_mm": max(item.event_total_max_mm for item in results),
        "maximum_reconstruction_difference_mm": max(
            item.max_total_difference_mm for item in results
        ),
        "total_output_bytes": sum(item.output_bytes for item in results),
        "normalization_scale_used_by_training_mm": 100.0,
    }
    manifest_path = args.output_dir / "time_slice_manifest.json"
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
