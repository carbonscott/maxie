#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse

parser = argparse.ArgumentParser(description='Fix epoch in a checkpoint file.')
parser.add_argument('path_checkpoint', help='Path to the checkpoint file.')
args = parser.parse_args()

path_checkpoint = args.path_checkpoint

checkpoint = torch.load(path_checkpoint, map_location = 'cpu')
checkpoint['training_state_dict']['epoch']+=1

path_checkpoint_output = f"{path_checkpoint}.patched"
torch.save(checkpoint, path_checkpoint_output)
