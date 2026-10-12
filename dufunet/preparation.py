"""Co-register and tile source rasters using Planet's pixel grid."""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm.auto import tqdm

from .constants import SENTINEL_BAND_NAMES


def _save(path, data, reference, window, *, nodata=None, descriptions=None):
    profile = reference.profile.copy()
    profile.update(driver='GTiff', count=data.shape[0], width=data.shape[2],
                   height=data.shape[1], transform=reference.window_transform(window),
                   dtype=data.dtype, compress='lzw', nodata=nodata)
    # Source JPEG compression/options can be incompatible with derived data.
    for option in ('photometric', 'interleave', 'predictor', 'blockxsize',
                   'blockysize', 'tiled'):
        profile.pop(option, None)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = tuple(descriptions)


def prepare_tiles(planet_path, mask_path, sentinel_path, output_root,
                  tile_size=256, stride=256, val_fraction=0.25,
                  min_label_fraction=0.10):
    """Split into contiguous left/train and right/val vertical strips.

    This is a new spatial split: it does not recreate the notebook's random
    tile split or establish a separate final test set.
    """
    root = Path(output_root)
    for split in ('train', 'val'):
        for kind in ('images', 'masks', 'sentinel'):
            (root / f'{split}_{kind}').mkdir(parents=True, exist_ok=True)
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    if not 0 <= min_label_fraction <= 1:
        raise ValueError('min_label_fraction must be between 0 and 1')
    if tile_size <= 0 or stride <= 0:
        raise ValueError('tile_size and stride must be positive')
    if tile_size != 256 or stride < tile_size:
        raise ValueError('Use 256-pixel tiles and non-overlapping stride for this training pipeline')

    manifest = []
    with rasterio.open(planet_path) as planet, rasterio.open(mask_path) as mask, \
         rasterio.open(sentinel_path) as sentinel:
        if planet.count != 3 or sentinel.count != 46 or mask.count != 1:
            raise ValueError('Expected Planet RGB=3, Sentinel=46 and mask=1 bands')
        columns = list(range(0, planet.width - tile_size + 1, stride))
        rows = list(range(0, planet.height - tile_size + 1, stride))
        if len(columns) < 2 or not rows:
            raise ValueError('The raster needs at least two tile columns and one row')
        split_col = min(len(columns)-1, max(1, round(len(columns)*(1-val_fraction))))
        # VRT transforms each input to exactly the Planet pixel grid.
        vrt = dict(crs=planet.crs, transform=planet.transform,
                   width=planet.width, height=planet.height)
        with WarpedVRT(mask, resampling=Resampling.nearest, **vrt) as mask_grid, \
             WarpedVRT(sentinel, resampling=Resampling.bilinear, **vrt) as sen_grid:
            for col_index, col in enumerate(tqdm(columns, desc='Tile columns')):
                split = 'train' if col_index < split_col else 'val'
                for row in rows:
                    window = Window(col, row, tile_size, tile_size)
                    rgb = planet.read(window=window)
                    labels = mask_grid.read(1, window=window)
                    valid_rgb = np.isfinite(rgb).all(axis=0)
                    if planet.nodata is not None:
                        valid_rgb &= (rgb != planet.nodata).all(axis=0)
                    if valid_rgb.mean() < 0.5 or np.isin(labels, range(1, 7)).mean() < min_label_fraction:
                        continue
                    features = sen_grid.read(window=window).astype(np.float32)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    stem = f'tile_r{row:07d}_c{col:07d}'
                    _save(root/f'{split}_images'/f'{stem}.tif', rgb, planet, window)
                    _save(root/f'{split}_masks'/f'{stem}.tif', labels[None], planet, window, nodata=0)
                    _save(root/f'{split}_sentinel'/f'{stem}.tif', features, planet, window,
                          descriptions=SENTINEL_BAND_NAMES)
                    manifest.append(dict(tile=stem, split=split, row=row, col=col))
    if not any(x['split']=='train' for x in manifest) or not any(x['split']=='val' for x in manifest):
        raise ValueError('One split has no valid tiles: adjust region or filters')
    (root/'split_manifest.json').write_text(json.dumps({
        'method': 'contiguous vertical strips, train left, validation right',
        'val_fraction_columns': val_fraction, 'split_column_index': split_col,
        'tile_size': tile_size, 'stride': stride, 'tiles': manifest,
    }, indent=2))
    return manifest


def calculate_stats(root, kind):
    """Compute train-only per-band means and stds from tile pixel values."""
    root = Path(root)
    paths = sorted((root/f'train_{kind}').glob('*.tif'))
    if not paths:
        raise ValueError(f'No train_{kind} tiles in {root}')
    n_bands = 46 if kind == 'sentinel' else 3
    sums = np.zeros(n_bands, dtype=np.float64)
    sums2 = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.float64)
    for path in tqdm(paths, desc=f'{kind} stats'):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float64)
        if data.shape[0] != n_bands:
            raise ValueError(f'Unexpected band count in {path}')
        for band in range(n_bands):
            values = data[band][np.isfinite(data[band])]
            sums[band] += values.sum()
            sums2[band] += np.square(values).sum()
            counts[band] += values.size
    if np.any(counts == 0):
        raise ValueError('A band has no finite training data')
    mean = sums / counts
    std = np.sqrt(np.maximum(0, sums2/counts - mean**2))
    std = np.maximum(std, 1e-6)
    names = SENTINEL_BAND_NAMES if kind == 'sentinel' else ['Red','Green','Blue']
    output = {'mean': mean.tolist(), 'std': std.tolist(), 'band_names': names}
    (root/f'{kind}_stats.json').write_text(json.dumps(output, indent=2))
    return output
