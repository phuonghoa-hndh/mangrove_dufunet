"""Co-registered Planet/Sentinel tile loading and train-time augmentation."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import rasterio
from .constants import (SENTINEL_BAND_NAMES, SEN_MEAN, SEN_STD,
                        PLANET_MEAN, PLANET_STD)

def _normalize_band_description(x):
    if x is None:
        return None
    x = str(x).strip()
    for token in x.replace('-', '_').split('_'):
        if token in SENTINEL_BAND_NAMES:
            return token
    return x


def _select_sentinel_indices(src):
    """Return 1-based source band indices for the configured Sentinel bands."""
    raw_count = src.count
    descriptions = list(src.descriptions) if src.descriptions else [None] * raw_count
    norm_desc = [_normalize_band_description(d) for d in descriptions]

    selected = []
    used = set()
    for band_name in SENTINEL_BAND_NAMES:
        if band_name in norm_desc:
            idx0 = norm_desc.index(band_name)
            if idx0 not in used:
                selected.append(idx0 + 1)
                used.add(idx0)

    if len(selected) != len(SENTINEL_BAND_NAMES):
        selected = list(range(1, 47))

    if len(selected) != len(SENTINEL_BAND_NAMES):
        raise ValueError('Expected 46 Sentinel bands')
    if max(selected) > raw_count:
        raise ValueError(
            f'Sentinel file has only {raw_count} bands, but selected indices are {selected}. '
            'Fix the input band order/count.'
        )
    return selected


class DualBranchDataset(Dataset):
    """
    Trả về dict:
      'planet'  : Tensor [3,  256, 256] float32 (normalized)
      'sentinel': Tensor [46, 256, 256] float32 (normalized)
      'labels'  : Tensor [256, 256]     int64   (0-5 or 255=ignore)
    """
    def __init__(self, img_paths, sen_paths, mask_paths,
                 planet_size=256, sentinel_size=256, augment=False, sentinel_mean=None, sentinel_std=None,
                 planet_mean=None, planet_std=None):
        assert len(img_paths) == len(sen_paths) == len(mask_paths)
        if planet_size != 256 or sentinel_size != 256:
            raise ValueError('Both Planet and Sentinel inputs must be 256x256.')
        self.img_paths     = img_paths
        self.sen_paths     = sen_paths
        self.mask_paths    = mask_paths
        self.planet_size   = planet_size
        self.sentinel_size = sentinel_size
        self.augment       = augment
        self.sen_mean = sentinel_mean if sentinel_mean is not None else SEN_MEAN
        self.sen_std = sentinel_std if sentinel_std is not None else SEN_STD
        self.planet_mean = planet_mean if planet_mean is not None else PLANET_MEAN
        self.planet_std = planet_std if planet_std is not None else PLANET_STD

    def __len__(self):
        return len(self.img_paths)

    def _read_planet(self, path):
        """GeoTIFF RGB → float32 tensor [3, H, W], normalized [0,1] per-image."""
        with rasterio.open(path) as src:
            arr = src.read([1,2,3]).astype(np.float32)
        p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
        arr = np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)
        return torch.from_numpy(arr)

    def _read_sentinel(self, path):
        """GeoTIFF → select 46 configured bands → normalized tensor [46, H, W]."""
        try:
            with rasterio.open(path) as src:
                selected = _select_sentinel_indices(src)
                arr = src.read(selected).astype(np.float32)

            if arr.shape[0] != len(SENTINEL_BAND_NAMES):
                raise ValueError(f'Sentinel selected shape is {arr.shape}, expected 46 channels.')

            mean = np.array(self.sen_mean, dtype=np.float32)[:, None, None]
            std  = np.array(self.sen_std,  dtype=np.float32)[:, None, None]
            arr  = (arr - mean) / (std + 1e-6)
            arr  = np.clip(arr, -3, 3)
            return torch.from_numpy(arr)
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading Sentinel file: {path} - {e}")
            raise # Re-raise the exception after printing the file path

    def _read_mask(self, path):
        """GeoTIFF mask → int64 tensor [H, W], values 0-5 (remap từ 1-6)."""
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int64)
        out = np.full_like(mask, 255)
        for v in range(1, 7):
            out[mask == v] = v - 1
        return torch.from_numpy(out)

    def _joint_augment(self, planet_t, sen_t, mask_t):
        """Augment giống nhau cho Planet/Sentinel/mask để giữ co-registration."""
        if random.random() > 0.5:
            planet_t = TF.hflip(planet_t)
            sen_t    = TF.hflip(sen_t)
            mask_t   = TF.hflip(mask_t)
        if random.random() > 0.5:
            planet_t = TF.vflip(planet_t)
            sen_t    = TF.vflip(sen_t)
            mask_t   = TF.vflip(mask_t)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            planet_t = torch.rot90(planet_t, k, dims=[1,2])
            sen_t    = torch.rot90(sen_t,    k, dims=[1,2])
            mask_t   = torch.rot90(mask_t,   k, dims=[0,1])
        if random.random() > 0.5:
            planet_t = TF.adjust_brightness(planet_t, 0.8 + random.random()*0.4)
            planet_t = TF.adjust_contrast(planet_t,   0.8 + random.random()*0.4)
        return planet_t, sen_t, mask_t

    def __getitem__(self, idx):
        planet_t = self._read_planet(self.img_paths[idx])
        sen_t    = self._read_sentinel(self.sen_paths[idx])
        mask_t   = self._read_mask(self.mask_paths[idx])

        planet_t = TF.resize(planet_t, [self.planet_size, self.planet_size],
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        sen_t    = TF.resize(sen_t, [self.sentinel_size, self.sentinel_size],
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask_t   = TF.resize(mask_t.unsqueeze(0), [self.planet_size, self.planet_size],
                             interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if self.augment:
            planet_t, sen_t, mask_t = self._joint_augment(planet_t, sen_t, mask_t)

        planet_t = TF.normalize(planet_t, mean=self.planet_mean, std=self.planet_std)
        return {'planet': planet_t, 'sentinel': sen_t, 'labels': mask_t.long()}


def paired_paths(root, split):
    """Return paired paths by stem; refuse partial/misordered triplets."""
    from pathlib import Path
    root = Path(root)
    groups = {kind: {p.stem: p for p in (root / f'{split}_{kind}').glob('*.tif')}
              for kind in ('images', 'sentinel', 'masks')}
    stems = set(groups['images'])
    if not stems or any(set(group) != stems for group in groups.values()):
        raise ValueError(f'Incomplete or unpaired {split} triplets under {root}')
    return tuple([[groups[kind][stem] for stem in sorted(stems)]
                  for kind in ('images', 'sentinel', 'masks')])
