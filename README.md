# WD-cDDPM

**Wet–dry-regularized conditional diffusion for event-maximum flood inundation mapping.**

WD-cDDPM predicts an event-maximum flood-depth field from terrain and spatially distributed, time-sliced rainfall. The implementation combines conditional denoising with wet–dry regularization to improve the delineation of inundation pathways and margins over mountainous terrain. Repeated sampling also provides ensemble depth statistics and wet probabilities.

## Model and scope

The conditioning tensor contains a DEM, a terrain-gradient magnitude field, and twelve consecutive ten-minute rainfall-depth fields: **14 conditioning channels** in total. The denoising network receives these fields together with the noisy target depth, giving **15 input channels** and one output channel.

The supplied configuration uses 256 × 256 training patches. Its wet–dry threshold is 0.05 m, and its regularization comprises binary cross-entropy, Dice, false-wet, and false-dry terms. The target is event-maximum depth, not a sequence of evolving flood states. The regularization does not directly impose the shallow-water equations or global mass conservation.

Reference labels in the associated study were generated using LISFLOOD-FP and Massflow, with each solver contributing half of the labeled cases; their depth fields were not averaged together. This repository contains the diffusion-model implementation, not the numerical solvers or the complete label-generation workflow.

## Repository contents

| Location | Purpose |
| --- | --- |
| `run.py`, `config/flood.json` | Training entry point and time-sliced rainfall configuration |
| `data/flood_dataset.py` | Paired terrain, rainfall, and maximum-depth loading |
| `models/model.py` | WDcDDPM training wrapper and exponential moving average |
| `models/network.py` | Conditional diffusion, reverse sampling, and wet–dry loss |
| `models/guided_diffusion_modules/` | Denoising U-Net and supporting operations |
| `models/sr3_modules/` | Retained alternative backbone implementation |
| `core/` | Configuration loading, logging, and training infrastructure |
| `inference_new_sample.py` | Single-case prediction, including overlapping tiles |
| `batch_test_inference.py` | Batch prediction and depth/extent metrics |
| `uq_test_single.py` | Ensemble sampling and uncertainty summaries |
| `scripts/` | Conversion of rainfall NetCDF inputs to ten-minute channels |

Datasets, trained weights, numerical-solver executables, manuscript figures, and deterministic baseline implementations are not bundled with this code copy. The commands below require separately prepared data and compatible weights.

## Installation

Use a Python 3.10+ environment. Install a compatible PyTorch/torchvision pair for your hardware, then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training and sampling. The requirements retain the source dependency ranges; they are not a pinned, independently tested environment specification. Run all commands from the repository root. No dataset or model download is triggered by the setup instructions.

## Input data

The default dataset layout is:

```text
datasets/flood/
├── terrain/
│   ├── dem_00.dem
│   └── dem_01.dem
├── rainfall/                         # optional NetCDF preprocessing inputs
│   ├── rainfall_seed_0.nc
│   └── rainfall_seed_1.nc
├── cumulative_rainfall_npy/
│   └── time_slice/
│       ├── cumulative_rain_seed_0.npy
│       └── cumulative_rain_seed_1.npy
└── results/
    ├── results_1/seed_1.max
    └── results_2/seed_2.max
```

For case index `i`, the loader pairs `dem_{i:02d}.dem` and `cumulative_rain_seed_{i}.npy` with `results_{i+1}/seed_{i+1}.max`. DEM and target files use six-line ASCII raster headers; elevations and depths are in metres. Rainfall arrays must have shape `[12, H, W]`, contain finite nonnegative values in **mm per ten-minute interval**, and align with the DEM. A two-dimensional event-total array is not a substitute for the twelve-channel input. Check the coordinate reference system, cell size, orientation, and grid alignment before preparing arrays; NumPy files do not carry georeferencing.

DEM and terrain-gradient magnitude are min–max normalized separately. The gradient field follows the source implementation's pixel-space `numpy.gradient` calculation. Rainfall uses a 100 mm per-channel normalization scale. Depth is square-root transformed, scaled by 3, and mapped to `[-1, 1]`, with clipping; this corresponds to a represented depth range of 0–9 m. The dataset loader replaces `-9999` raster cells with zero, removes target depths below 0.05 m, and excludes cases whose maximum reference depth exceeds 30 m. Prepare a valid evaluation mask separately if the data contain no-data regions.

### Rainfall preprocessing

For one event:

```sh
python scripts/convert_rainfall_netcdf_to_timeslices.py --input datasets/flood/rainfall/rainfall_seed_0.nc --dem datasets/flood/terrain/dem_00.dem --output datasets/flood/cumulative_rainfall_npy/time_slice/cumulative_rain_seed_0.npy
```

This converter expects `rainfall_depth` in mm and `time` in hours at one-minute intervals. It sums consecutive ten-minute windows, checks the spatial dimensions against the DEM, and writes a float32 array and a conversion record. For the model input, supply a 120-minute sequence to obtain twelve channels. This is a converter for the project's NetCDF forcing format, not a general satellite-rainfall reader or a temporal downscaling method.

`scripts/build_time_slice_rainfall.py` provides batch conversion with optional checks against existing event-total arrays. Use `--help` to inspect its input/output directory options.

## Training

Edit `config/flood.json` to set both dataset roots, the requested sample range, validation split, and hardware options. The `data_root` entries point to the **results subdirectory**, unlike `--data-root` in the batch scripts, which points to its parent. The supplied configuration requests 6,000 indexed cases and reserves 32 valid pairs for validation; adjust these settings to your available development data. They do not automatically establish the terrain-held-out experimental splits described in the manuscript.

```sh
python run.py -p train -c config/flood.json
```

Training uses a learning rate of 5 × 10⁻⁵, a batch size of 8, an EMA decay of 0.9999, and a maximum of 570 epochs. Training diffusion has 2,000 steps. Checkpoints and logs are written under `experiments/`. The configuration supports `//` comments through the project's configuration parser.

## Single-case prediction

Use a compatible 15-input-channel network checkpoint containing a network state dictionary, such as a saved `*_Network_ema.pth`. Supply paths explicitly; local historical example names retained in the script are not bundled sample data.

```sh
python inference_new_sample.py --dem datasets/flood/terrain/dem_00.dem --rain datasets/flood/cumulative_rainfall_npy/time_slice/cumulative_rain_seed_0.npy --weights checkpoints/wd_cddpm_ema.pth --output-dir outputs/single_case --device cuda
```

No reference depth is needed for this command. By default, the script predicts overlapping 256 × 256 tiles with 32-pixel overlap and averages their normalized predictions, allowing a larger raster without resizing it. `--tile-size 0` selects full-grid inference and can require substantially more GPU memory. The single-case script retains the source implementation's **1,000-step** sampling schedule and seed 1024. The output is a depth `.npy` array and a diagnostic PNG; it is not a georeferenced GeoTIFF. Output basenames retain the original example identifiers (`dem_999_rain_998`), so use a separate output directory for each case.

## Batch evaluation

```sh
python batch_test_inference.py --data-root datasets/flood --start 0 --end 9 --weights checkpoints/wd_cddpm_ema.pth --output-dir outputs/batch --test-steps 200 --cell-size 8.35 --device cuda --no-save-figs
```

`--start` and `--end` are inclusive terrain/rainfall indices, with the same `i` to `i+1` target-file mapping described above. Set `--cell-size` to the actual grid spacing in metres for area metrics. The script writes predictions and CSV metrics, including RMSE, CSI, and area/depth errors. It uses a 0.05 m wet threshold. Although a legacy `--median-size` option remains, the actual median-filter call is commented out in this source version, so no median smoothing is applied. `--save-raw` additionally saves the intermediate depth array. `--no-save-figs` skips diagnostic figures without changing the evaluation.

## Ensemble sampling

```sh
python uq_test_single.py --data-root datasets/flood --indices 0 --n-samples 30 --weights checkpoints/wd_cddpm_ema.pth --output-dir outputs/ensemble --test-steps 200 --cell-size 8.35 --device cuda --no-save-figs
```

This script uses the batch utility functions and requires the corresponding reference files. It generates ensemble depth summaries and wet probabilities, with optional member saving via `--save-members`. Its default sampling schedule has 1,000 steps; the example explicitly selects 200. Keep sampling schedules, seeds, thresholds, and filtering consistent when comparing results. Ensemble spread describes variability in stochastic model sampling, not a complete accounting of rainfall, terrain, or observation uncertainty.

## Code basis and license

WD-cDDPM builds on the training infrastructure of [Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) and the guided-diffusion U-Net backbone, with adaptations for flood-depth prediction, time-sliced rainfall conditioning, and wet–dry regularization. The original MIT license and copyright notice are retained in `LICENSE`, together with source attribution comments.
