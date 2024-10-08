import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Optional, Union, List, Tuple

from math import ceil

## from ..perf import Timer

import logging
logger = logging.getLogger(__name__)

@dataclass
class DummyImageDataConfig:
    C          : int
    H          : int
    W          : int
    sample_size: int


class DummyImageData(Dataset):
    def __init__(self, config):
        self.config = config

    def __getitem__(self, idx):
        C           = self.config.C
        H           = self.config.H
        W           = self.config.W
        sample_size = self.config.sample_size

        input = torch.randn(C, H, W)

        return input

    def __len__(self):
        return self.config.sample_size


@dataclass
class DistributedSegmentedDummyImageDataConfig:
    C              : int
    H              : int
    W              : int
    seg_size       : int
    total_size     : int
    dist_rank      : int
    dist_world_size: int
    transforms     : Union[None, List, Tuple]
    dtype          : torch.dtype


class DistributedSegmentedDummyImageData(Dataset):
    def __init__(self, config):
        self.config = config

        self.total_size = config.total_size
        self.seg_size   = config.seg_size
        self.transforms = config.transforms
        self.dtype      = config.dtype

        self.start_idx   = 0
        self.end_idx     = 0
        self.current_dataset = None

    def reset(self):
        self.start_idx       = 0
        self.end_idx         = 0
        self.current_dataset = None

    @property
    def num_seg(self):
        return ceil(self.config.total_size / (self.config.seg_size * self.config.dist_world_size))

    def __getitem__(self, idx):
        global_idx = self.current_dataset[idx]

        ## logger.debug(f"[RANK {self.config.dist_rank}] DATA IDX = {global_idx}")
        ## print(f"[RANK {self.config.dist_rank}] DATA IDX = {global_idx}")

        C = self.config.C
        H = self.config.H
        W = self.config.W

        input = torch.randn(1, C, H, W)  # B=1

        # Apply transformation to input and label at the same time...
        if self.transforms is not None:
            if self.dtype is not None: input = input.to(self.dtype)
            for enum_idx, trans in enumerate(self.transforms):
                input = trans(input)

        return input[0]  # Remove the B, which will be handled by pytorch's dataloader

    def __len__(self):
        return self.end_idx - self.start_idx

    def calculate_end_idx(self):
        """
        end_idx is not inclusive (up to, but not including end_idx)
        """
        # Calculate and return the end index for the current dataset segment.
        return min(self.start_idx + self.config.seg_size * self.config.dist_world_size, self.config.total_size)

    def update_dataset_segment(self):
        logger.debug(f"[RANK {self.config.dist_rank}] Updating segment to {self.start_idx}-{self.end_idx}.")
        return list(range(self.start_idx, self.end_idx))

    def set_start_idx(self, start_idx):
        requires_reset = False

        logger.debug(f"[RANK {self.config.dist_rank}] Setting start idx to {start_idx}.")

        self.start_idx = start_idx
        self.end_idx   = self.calculate_end_idx()

        # Update dataset segment and sync across ranks
        object_list = [None,]  # For communication
        if self.config.dist_rank == 0:
            self.current_dataset = self.update_dataset_segment()
            object_list = [self.current_dataset,]

        if self.config.dist_world_size > 1:
            logger.debug(f"[RANK {self.config.dist_rank}] Syncing current dataset.")
            dist.broadcast_object_list(object_list, src = 0)
            self.current_dataset = object_list[0]

        # Reset if reached the end of the item generator???
        if len(self.current_dataset) == 0:
            requires_reset = True
            self.reset()

        return requires_reset


