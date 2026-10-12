"""Compare CRAM/Concat predictions using the shared training model/dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from dufunet.dataset import DualBranchDataset, paired_paths
from dufunet.model import DualBranchFusion
from dufunet.training import upsample
from dufunet.visualization import colorize,error_map
from dufunet.constants import PLANET_MEAN,PLANET_STD


def load_model(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=DualBranchFusion(ckpt['cfg'],pretrained=False)
    model.load_state_dict(ckpt['model'],strict=True)
    model.to(device).eval()
    return model,ckpt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--concat',type=Path,required=True)
    p.add_argument('--cram',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/figures'))
    p.add_argument('--tiles',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--sentinel-stats',type=Path,
                   help='Supply the same 46-band stats used for both checkpoints')
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    concat,cc=load_model(args.concat,device)
    cram,cr=load_model(args.cram,device)
    normalization=[ckpt.get('normalization',{}).get('sentinel','notebook constants')
                   for ckpt in (cc,cr)]
    if bool(args.sentinel_stats) != any(v!='notebook constants' for v in normalization):
        raise ValueError('Provide --sentinel-stats if and only if checkpoints used custom stats')
    if len(set(normalization)) > 1:
        raise ValueError('Checkpoints used different Sentinel normalization')
    stats=json.loads(args.sentinel_stats.read_text()) if args.sentinel_stats else {}
    ds=DualBranchDataset(*paired_paths(args.data,'val'),augment=False,
                         sentinel_mean=stats.get('mean'),sentinel_std=stats.get('std'))
    if any(i<0 or i>=len(ds) for i in args.tiles):
        raise ValueError(f'Tile index out of range: 0..{len(ds)-1}')
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(args.tiles),6,figsize=(15,2.8*len(args.tiles)),squeeze=False)
    names=['Planet RGB','Ground truth','Concat','Concat error','CRAM','CRAM error']
    for ax,name in zip(axes[0],names):ax.set_title(name)
    for row,i in enumerate(args.tiles):
        sample=ds[i]
        gt=sample['labels'].numpy()
        rgb=sample['planet'].numpy()*np.array(PLANET_STD)[:,None,None]+np.array(PLANET_MEAN)[:,None,None]
        rgb=rgb.clip(0,1).transpose(1,2,0)
        with torch.inference_mode():
            x=sample['planet'][None].to(device);y=sample['sentinel'][None].to(device)
            pred_c=upsample(concat(x,y),gt.shape).argmax(1)[0].cpu().numpy()
            pred_r=upsample(cram(x,y),gt.shape).argmax(1)[0].cpu().numpy()
        panels=(rgb,colorize(gt),colorize(pred_c),error_map(pred_c,gt),
                colorize(pred_r),error_map(pred_r,gt))
        for ax,im in zip(axes[row],panels):
            ax.imshow(im);ax.axis('off')
        axes[row,0].set_ylabel(f'Tile {i}')
    fig.tight_layout()
    target=args.output/'fusion_qualitative_comparison.png'
    fig.savefig(target,dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(target)


if __name__=='__main__':
    main()
