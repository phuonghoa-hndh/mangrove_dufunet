# DuFuNet mangrove mapping — Python project

A Python project extracted and reorganized from the three supplied notebooks. The original model definitions and training formulas are in `dufunet/`. Command line programs are in `scripts/`. Data and checkpoints are **not** included.

```text
mangrove-dufunet-python/
├── dufunet/
│   ├── constants.py       # class labels, band order, original normalization
│   ├── preparation.py     # co-register rasters, spatial tile split, stats
│   ├── dataset.py         # paired tile loading and augmentations
│   ├── model.py           # SegFormer-B2, EfficientNet-B0, CRAM/Concat
│   ├── losses.py          # focal and Dice loss
│   ├── metrics.py         # OA, macro IoU, macro F1
│   ├── training.py        # seeding and training/validation epoch
│   └── visualization.py   # class colors and error map
├── scripts/
│   ├── prepare_data.py
│   ├── validate_tiles.py
│   ├── train.py
│   ├── aggregate.py
│   └── qualitative.py
├── data/raw/              # imagery, masks (ignored by Git)
├── data/processed/        # six tile directories (ignored by Git)
├── outputs/               # checkpoints, metrics, figures (ignored)
├── docs/notes.md
├── requirements.txt
└── .gitignore
```

## Setup

Run commands from the project root. Install a compatible `torch` and `torchvision` pair for your machine, then:

```bash
python -m venv .venv
# Activate the environment for your operating system.
pip install -r requirements.txt
```

For the original Kaggle/Colab environments, their compatible PyTorch builds may already be installed. The SegFormer and EfficientNet pretrained weights are fetched when training starts, so the first training run needs access to those model weights.

## Prepare data

Supply Planet RGB, a one-band label raster with class codes 1–6 (0 = ignored), and a 46-band Sentinel feature stack in the order listed in `dufunet/constants.py`.

```bash
python -m scripts.prepare_data \
  --planet data/raw/planet.tif \
  --mask data/raw/labels.tif \
  --sentinel data/raw/sentinel_46.tif \
  --output data/processed
python -m scripts.validate_tiles data/processed
```

Preparation reprojects mask with nearest-neighbor and Sentinel with bilinear sampling onto the Planet pixel grid. It cuts non-overlapping 256-pixel tiles. By default, the left 75% of tile columns forms training and the right 25% forms validation; `split_manifest.json` records tile coordinates. This **new split** is not the notebook's original random split and is not a held-out final test region. Examine the region and mask coverage before treating it as an appropriate spatial evaluation. Train-only raw-value `sentinel_stats.json` and `images_stats.json` are also saved. Resampling does not improve the native Sentinel resolution.

The resulting directory has `train_images`, `train_masks`, `train_sentinel`, `val_images`, `val_masks`, and `val_sentinel` folders with matching `.tif` stems.

## Train and compare

The default Sentinel normalization is the **hardcoded constants from the training notebook**, which helps preserve its training setup. To use the new train-split statistics, add `--sentinel-stats data/processed/sentinel_stats.json` to **both** training commands and qualitative analysis. Do not compare checkpoints trained with different normalization.

```bash
python -m scripts.train --data data/processed --mode concat --seed 42
python -m scripts.train --data data/processed --mode cram --seed 42
python -m scripts.aggregate --metrics outputs/metrics
python -m scripts.qualitative --data data/processed \
  --concat outputs/checkpoints/F1-Concat_seed42_best.pth \
  --cram outputs/checkpoints/F1-CRAM_seed42_best.pth \
  --tiles 0 1 2 3
```

To run the five seeds listed in the notebook, repeat both modes with seeds `42`, `123`, `2026`, `3407`, and `2025`. Each command trains one seed and writes a best-validation checkpoint, per-epoch history, and per-seed metrics. The qualitative command uses strict checkpoint loading and the training preprocessing; it produces a 2-model comparison. The original 5-model figure also requires Planet-only and Sentinel-only baseline checkpoints, which were not supplied.

## Limits

This project passed syntax checks, module structure checks and source-equivalence checks for the extracted model/loss definitions. Training and raster processing could not be run in this workspace because PyTorch and rasterio were unavailable and no source rasters/checkpoints were supplied. Read [`docs/notes.md`](docs/notes.md) for differences from the notebooks and study reporting requirements.

## GitHub

Create an empty repository, then from this folder:

```bash
git init
git add .
git commit -m "Organize DuFuNet Python pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Choose a license only after confirming redistribution rights for the research code and data.
