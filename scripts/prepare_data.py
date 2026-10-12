"""Prepare aligned tiles and train-only statistics."""
import argparse
from dufunet.preparation import prepare_tiles, calculate_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planet', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--sentinel', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--stride', type=int, default=256)
    p.add_argument('--val-fraction', type=float, default=0.25)
    args = p.parse_args()
    tiles = prepare_tiles(args.planet, args.mask, args.sentinel, args.output,
                          tile_size=args.tile_size, stride=args.stride,
                          val_fraction=args.val_fraction)
    calculate_stats(args.output, 'sentinel')
    calculate_stats(args.output, 'images')
    print(f'Wrote {len(tiles)} spatially split tile triplets to {args.output}')


if __name__ == '__main__':
    main()
