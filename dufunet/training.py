"""Training utilities adapted from notebook cells 16, 20-27."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import SegMetrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def upsample(logits, size):
    return F.interpolate(logits, size=size, mode='bilinear', align_corners=False)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metrics = SegMetrics(6)
    for batch in loader:
        planet = batch['planet'].to(device)
        sentinel = batch['sentinel'].to(device)
        labels = batch['labels'].to(device)
        with torch.set_grad_enabled(training):
            logits = upsample(model(planet, sentinel), labels.shape[-2:])
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        total_loss += loss.item()
        metrics.update(logits.argmax(1).cpu().numpy(), labels.cpu().numpy())
    if not len(loader):
        raise ValueError('Empty training/validation loader')
    return total_loss/len(loader), metrics.compute()
