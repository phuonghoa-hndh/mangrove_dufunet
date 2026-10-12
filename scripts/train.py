"""Train one fusion mode and seed; saves best validation mIoU checkpoint."""
import argparse
import json
import time
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dufunet.constants import SENTINEL_BAND_NAMES
from dufunet.dataset import DualBranchDataset, paired_paths
from dufunet.losses import CombinedLoss
from dufunet.model import DualBranchFusion
from dufunet.training import run_epoch, seed_everything


def seed_worker(worker_id, base_seed):
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def config(mode, seed):
    return dict(fusion_mode=mode,
                experiment_id='F1-CRAM' if mode=='cram' else 'F1-Concat',
                planet_backbone='nvidia/mit-b2', planet_channels=3,
                sentinel_backbone='efficientnet_b0', sentinel_channels=46,
                sentinel_band_names=SENTINEL_BAND_NAMES,
                image_size=256, sentinel_size=256, cram_dim=256,
                cram_heads=8, num_classes=6, loss_type='focal+dice',
                focal_gamma=2.0, class_weights=[2.26,1.45,1.79,0.52,1.56,1.23],
                epochs=80, batch_size=4, lr_planet_enc=6e-5,
                lr_sentinel_enc=1e-4, lr_align_decoder=6e-4,
                weight_decay=0.01, warmup_epochs=5, use_augment=True, seed=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--mode', choices=['cram','concat'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--sentinel-stats', type=Path,
                   help='Optional stats JSON; omit to use the notebook training constants')
    args = p.parse_args()
    cfg = config(args.mode, args.seed)
    cfg['epochs'], cfg['batch_size'] = args.epochs, args.batch_size
    if args.epochs <= cfg['warmup_epochs']:
        raise ValueError('epochs must exceed the 5 warmup epochs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['device'] = device
    seed_everything(args.seed)
    stats = json.loads(args.sentinel_stats.read_text()) if args.sentinel_stats else {}
    if stats and (len(stats['mean']) != 46 or len(stats['std']) != 46):
        raise ValueError('Sentinel stats must contain 46 mean/std entries')
    kwargs = dict(planet_size=256, sentinel_size=256,
                  sentinel_mean=stats.get('mean'), sentinel_std=stats.get('std'))
    train = DualBranchDataset(*paired_paths(args.data,'train'), augment=True, **kwargs)
    val = DualBranchDataset(*paired_paths(args.data,'val'), augment=False, **kwargs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device=='cuda'),
                              generator=generator,
                              worker_init_fn=partial(seed_worker, base_seed=args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device=='cuda'))
    model = DualBranchFusion(cfg, pretrained=True).to(device)
    criterion = CombinedLoss(gamma=cfg['focal_gamma'],
                             weights=torch.tensor(cfg['class_weights'], dtype=torch.float32))
    optimizer = AdamW([
        {'params':model.planet_encoder.parameters(), 'lr':cfg['lr_planet_enc']},
        {'params':model.sentinel_encoder.parameters(), 'lr':cfg['lr_sentinel_enc']},
        {'params':list(model.aligner.parameters())+list(model.fusion.parameters())+
                  list(model.decoder.parameters()), 'lr':cfg['lr_align_decoder']},
    ], weight_decay=cfg['weight_decay'])
    scheduler = OneCycleLR(optimizer,
        max_lr=[cfg['lr_planet_enc'],cfg['lr_sentinel_enc'],cfg['lr_align_decoder']],
        total_steps=args.epochs*len(train_loader),
        pct_start=cfg['warmup_epochs']/args.epochs, anneal_strategy='cos',
        div_factor=25, final_div_factor=1e4)
    exp = f"{cfg['experiment_id']}_seed{args.seed}"
    for name in ('checkpoints','metrics'):
        (args.output/name).mkdir(parents=True, exist_ok=True)
    path = args.output/'checkpoints'/f'{exp}_best.pth'
    history=[]; best=-1.0
    for epoch in range(1,args.epochs+1):
        start=time.time()
        tr_loss,tr=run_epoch(model,train_loader,criterion,device,optimizer,scheduler)
        va_loss,va=run_epoch(model,val_loader,criterion,device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            train_mIoU=tr['mIoU'], val_mIoU=va['mIoU'],
                            val_mF1=va['mF1'], val_OA=va['OA'],
                            val_F1=va['F1'], val_IoU=va['IoU'],
                            time_s=time.time()-start))
        if va['mIoU']>best:
            best=va['mIoU']
            torch.save(dict(epoch=epoch,model=model.state_dict(),best_miou=best,
                            cfg=cfg, normalization={'sentinel':str(args.sentinel_stats) if args.sentinel_stats else 'notebook constants'}),path)
        (args.output/'metrics'/f'{exp}_history.json').write_text(json.dumps(history,indent=2))
        print(f'{exp} epoch={epoch} train={tr_loss:.4f} val={va_loss:.4f} mIoU={va["mIoU"]:.4f}')
    ckpt=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt['model'],strict=True)
    _,final=run_epoch(model,val_loader,criterion,device)
    result=dict(model=cfg['experiment_id'],fusion_mode=args.mode,seed=args.seed,
                checkpoint=str(path),best_epoch=ckpt['epoch'],**final)
    (args.output/'metrics'/f'{exp}_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
