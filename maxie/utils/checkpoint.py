#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path):
    torch.save({
        'epoch'               : epoch,
        'loss_min'            : loss_min,
        'model_state_dict'    : model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location = device)
    if model     is not None: model.module.load_state_dict(checkpoint['model_state_dict']) \
                              if hasattr(model, 'module') else \
                              model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['loss_min']
