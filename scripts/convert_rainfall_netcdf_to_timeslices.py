"""Convert one NetCDF rainfall event into 10-minute rainfall channels.

The diffusion flood model consumes ``[12, H, W]`` arrays.  LISFLOOD NetCDF
files store ``rainfall_depth`` as ``[time, y, x]`` and the time values are in
hours.  This utility validates the one-minute cadence, aligns the spatial axes
to an ASCII DEM, sums consecutive ten-minute windows, and writes a float32
array.  A two-dimensional event-total array can also be written for the
existing cumulative-rainfall convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def read_ascii_shape(path: Path) -> tuple[int, int]:
    """Read the dimensions from a six-line LISFLOOD ASCII raster header."""
    header = {}
    with path.open("r", encoding="ascii") as handle:
        for _ in range(6):
            key, value = handle.readline().split()[:2]
            header[key.lower()] = float(value)
    rows = int(header["nrows"])
    cols = int(header["ncols"])
    return rows, cols


def convert(
    input_path: Path,
    output_path: Path,
    dem_path: Path,
    window_minutes: int = 10,
    cumulative_output: Path | None = None,
) -> dict:
    dem_shape = read_ascii_shape(dem_path)

    with Dataset(str(input_path), mode="r") as dataset:
        if "time" not in dataset.variables or "rainfall_depth" not in dataset.variables:
            raise ValueError("NetCDF must contain time and rainfall_depth variables")
        rainfall_variable = dataset.variables["rainfall_depth"]
        time_variable = dataset.variables["time"]
        if getattr(rainfall_variable, "units", None) != "mm":
            raise ValueError(f"rainfall_depth units must be mm, got {getattr(rainfall_variable, 'units', None)!r}")
        if getattr(time_variable, "units", None) != "hour":
            raise ValueError(f"time units must be hour, got {getattr(time_variable, 'units', None)!r}")

        rainfall = np.ma.filled(rainfall_variable[:], np.nan).astype(np.float32)
        time = np.asarray(time_variable[:], dtype=np.float64)
        variable_dimensions = tuple(rainfall_variable.dimensions)

    if rainfall.ndim != 3 or rainfall.shape[0] != time.size:
        raise ValueError(f"rainfall_depth must be [time, height, width], got {rainfall.shape}")
    if not np.isfinite(rainfall).all() or np.any(rainfall < 0.0):
        raise ValueError("rainfall_depth contains NaN, infinity, or negative values")
    if time.size < 2:
        raise ValueError("At least two time coordinates are required")

    step_minutes = np.diff(time) * 60.0
    if not np.allclose(step_minutes, 1.0, atol=1.0e-3, rtol=0.0):
        raise ValueError(f"Expected one-minute time steps, got {step_minutes[:5]} minutes")
    if window_minutes <= 0 or rainfall.shape[0] % window_minutes:
        raise ValueError(
            f"{rainfall.shape[0]} time steps cannot be divided into {window_minutes}-minute windows"
        )

    raw_spatial_shape = tuple(rainfall.shape[1:])
    if raw_spatial_shape == dem_shape:
        aligned = rainfall
        spatial_transform = "none"
    elif raw_spatial_shape[::-1] == dem_shape:
        # The supplied 999 file declares x=938 and y=898 while the DEM is
        # nrows=938, ncols=898.  Its [y, x] data therefore need transposing.
        aligned = rainfall.transpose(0, 2, 1)
        spatial_transform = "transpose_yx_to_dem_rows_cols"
    else:
        raise ValueError(
            f"Rainfall spatial shape {raw_spatial_shape} does not match DEM shape {dem_shape}"
        )

    channels = aligned.reshape(
        aligned.shape[0] // window_minutes,
        window_minutes,
        dem_shape[0],
        dem_shape[1],
    ).sum(axis=1, dtype=np.float32)
    if channels.shape[0] != 12:
        raise ValueError(f"Expected 12 ten-minute channels, got {channels.shape[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, channels.astype(np.float32, copy=False), allow_pickle=False)
    cumulative = channels.sum(axis=0, dtype=np.float32)
    if cumulative_output is not None:
        cumulative_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(cumulative_output, cumulative, allow_pickle=False)

    reconstructed = channels.sum(axis=0, dtype=np.float32)
    source_total = aligned.sum(axis=0, dtype=np.float32)
    max_difference = float(np.max(np.abs(reconstructed - source_total)))
    if not np.allclose(reconstructed, source_total, atol=1.0e-4, rtol=1.0e-6):
        raise ValueError(f"Rainfall total is not conserved; max difference is {max_difference}")

    manifest = {
        # ``Path.resolve`` can require a Windows filesystem permission on some
        # mounted workspaces; ``absolute`` is sufficient for this manifest.
        "source": str(input_path.absolute()),
        "dem": str(dem_path.absolute()),
        "output": str(output_path.absolute()),
        "cumulative_output": str(cumulative_output.absolute()) if cumulative_output else None,
        "source_variable": "rainfall_depth",
        "source_dimensions": variable_dimensions,
        "source_shape": list(rainfall.shape),
        "dem_shape": list(dem_shape),
        "spatial_transform": spatial_transform,
        "source_units": "mm per one-minute step",
        "output_units": "mm per ten-minute window",
        "output_shape": list(channels.shape),
        "window_minutes": window_minutes,
        "total_rainfall_mm": float(cumulative.max()),
        "maximum_reconstruction_difference_mm": max_difference,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--cumulative-output", type=Path, default=None)
    parser.add_argument("--window-minutes", type=int, default=10)
    args = parser.parse_args()
    manifest = convert(
        input_path=args.input,
        output_path=args.output,
        dem_path=args.dem,
        window_minutes=args.window_minutes,
        cumulative_output=args.cumulative_output,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
