"""Aggregate saved per-seed metrics; run seeds before reporting uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,default=Path('outputs/metrics'))
    args=p.parse_args()
    rows=[json.loads(path.read_text()) for path in args.metrics.glob('*_metrics.json')]
    if not rows:
        raise ValueError(f'No per-seed metrics in {args.metrics}')
    summary={}
    for mode in sorted({row['fusion_mode'] for row in rows}):
        runs=[r for r in rows if r['fusion_mode']==mode]
        seeds=[r['seed'] for r in runs]
        if len(set(seeds))!=len(seeds):
            raise ValueError(f'Duplicate seeds for {mode}')
        summary[mode]={'seeds':seeds,'n':len(runs)}
        for metric in ('mIoU','mF1','OA'):
            values=np.array([r[metric] for r in runs],dtype=float)
            summary[mode][metric]={'mean':float(values.mean()),
                'sd':float(values.std(ddof=1)) if len(values)>1 else None}
    target=args.metrics/'summary.json'
    target.write_text(json.dumps(summary,indent=2))
    print(target.read_text())


if __name__=='__main__':
    main()
