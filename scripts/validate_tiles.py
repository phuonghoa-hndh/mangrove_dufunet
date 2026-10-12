"""Check paired tile stems, shapes, CRS, band counts and label codes."""
import argparse
from pathlib import Path
import numpy as np
import rasterio


def validate(root,split):
    groups={kind:{p.stem:p for p in (root/f'{split}_{kind}').glob('*.tif')}
            for kind in ('images','sentinel','masks')}
    stems=set(groups['images'])
    if not stems or any(set(files)!=stems for files in groups.values()):
        raise ValueError(f'{split}: missing or mismatched tile stems')
    for stem in sorted(stems):
        with rasterio.open(groups['images'][stem]) as planet, \
             rasterio.open(groups['sentinel'][stem]) as sentinel, \
             rasterio.open(groups['masks'][stem]) as mask:
            if (planet.count,sentinel.count,mask.count)!=(3,46,1):
                raise ValueError(f'{split}/{stem}: expected bands 3/46/1')
            if len({(src.width,src.height,src.crs,src.transform) for src in
                    (planet,sentinel,mask)})!=1:
                raise ValueError(f'{split}/{stem}: pixel grids differ')
            if not np.isin(mask.read(1),range(7)).all():
                raise ValueError(f'{split}/{stem}: labels outside 0..6')
    return stems


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    args=p.parse_args()
    train=validate(args.data,'train');val=validate(args.data,'val')
    if train & val:raise ValueError('Overlapping tile stems between splits')
    print(f'OK: {len(train)} train and {len(val)} validation triplets')


if __name__=='__main__':
    main()
