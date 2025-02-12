from typing import Tuple, Optional, List, Union
from pydantic import BaseModel
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from math import ceil
import logging

from maxie.tensor_transforms import (
    NoTransform,
    PolarCenterCrop,
    MergeBatchPatchDims,
    Pad,
    InstanceNorm,
    RandomPatch,
    RandomRotate,
    RandomShift,
)

logger = logging.getLogger(__name__)

class TransformationPipeline(BaseModel):
    """Container for pre-transforms and runtime transforms"""
    pre_transforms: Tuple = ()
    transforms: Tuple = ()
    merges_batch_patch_dims: bool = False

    class Config:
        arbitrary_types_allowed = True

class DummyDataConfig(BaseModel):
    """Configuration for dummy dataset"""
    C: int
    H: int
    W: int
    seg_size: int
    total_size: int
    dist_rank: int = 0
    dist_world_size: int = 1
    transforms: Optional[Union[List, Tuple]] = None
    dtype: torch.dtype = torch.float32

    class Config:
        arbitrary_types_allowed = True

class DistributedSegmentedDummyData(Dataset):
    """Dummy dataset that mimics distributed segmented data loading"""

    def __init__(self, config: DummyDataConfig):
        self.config = config
        self.total_size = config.total_size
        self.seg_size = config.seg_size
        self.transforms = config.transforms
        self.dtype = config.dtype

        self.start_idx = 0
        self.end_idx = 0
        self.current_dataset = None

        # Initialize the first segment
        self.set_start_idx(0)

    def reset(self):
        """Reset dataset state"""
        self.start_idx = 0
        self.end_idx = 0
        self.current_dataset = None

    @property
    def num_seg(self):
        """Calculate number of segments"""
        return ceil(self.config.total_size / (self.config.seg_size * self.config.dist_world_size))

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single item from the dataset"""
        global_idx = self.current_dataset[idx]

        input = torch.randn(1, self.config.C, self.config.H, self.config.W)  # B=1

        if self.transforms is not None:
            if self.dtype is not None:
                input = input.to(self.dtype)
            for trans in self.transforms:
                input = trans(input)

        return input[0]  # Remove the batch dimension

    def __len__(self) -> int:
        """Get dataset length"""
        return self.end_idx - self.start_idx

    def calculate_end_idx(self) -> int:
        """Calculate end index for current segment"""
        return min(
            self.start_idx + self.config.seg_size * self.config.dist_world_size,
            self.config.total_size
        )

    def update_dataset_segment(self) -> List[int]:
        """Update the current dataset segment"""
        logger.debug(f"[RANK {self.config.dist_rank}] Updating segment to {self.start_idx}-{self.end_idx}.")
        return list(range(self.start_idx, self.end_idx))

    def set_start_idx(self, start_idx: int) -> bool:
        """Set the starting index for the dataset segment"""
        requires_reset = False

        logger.debug(f"[RANK {self.config.dist_rank}] Setting start idx to {start_idx}.")

        self.start_idx = start_idx
        self.end_idx = self.calculate_end_idx()

        # Update dataset segment and sync across ranks
        object_list = [None]
        if self.config.dist_rank == 0:
            self.current_dataset = self.update_dataset_segment()
            object_list = [self.current_dataset]

        if self.config.dist_world_size > 1:
            logger.debug(f"[RANK {self.config.dist_rank}] Syncing current dataset.")
            dist.broadcast_object_list(object_list, src=0)
            self.current_dataset = object_list[0]

        if len(self.current_dataset) == 0:
            requires_reset = True
            self.reset()

        return requires_reset

class DatasetManager:
    """Manages dataset creation and transformations for training and evaluation"""

    def __init__(self, config: 'TrainingConfig', dist_env: dict):
        self.config = config
        self.dist_env = dist_env
        self.base_seed = 0

    def setup_transforms(self) -> TransformationPipeline:
        """Set up all data transformations based on configuration"""
        transforms_config = self.config.dataset.transforms
        transform_set = transforms_config.set

        # Pre-transforms setup
        pre_transforms = []
        if transform_set.pad:
            pre_transforms.append(Pad(
                transforms_config.H_pad,
                transforms_config.W_pad
            ))

        if transform_set.polar_center_crop:
            pre_transforms.append(PolarCenterCrop(
                Hv=transforms_config.Hv,
                Wv=transforms_config.Wv,
                sigma=transforms_config.sigma,
                num_crop=transforms_config.num_crop,
            ))

        merges_batch_patch_dims = transform_set.polar_center_crop
        if merges_batch_patch_dims:
            pre_transforms.append(MergeBatchPatchDims())

        if not pre_transforms:
            pre_transforms.append(NoTransform())

        # Runtime transforms setup
        transforms = []
        if transform_set.uses_instance_norm:
            transforms.append(InstanceNorm())

        if transform_set.random_rotate:
            transforms.append(RandomRotate(
                angle_max=transforms_config.angle_max
            ))

        if transform_set.random_shift:
            transforms.append(RandomShift(
                frac_y_shift_max=transforms_config.frac_shift_max,
                frac_x_shift_max=transforms_config.frac_shift_max,
            ))

        if transform_set.random_patch:
            transforms.append(RandomPatch(
                num_patch=transforms_config.num_patch,
                size_patch=transforms_config.size_patch,
                var_size_patch=transforms_config.var_size_patch
            ))

        if not transforms:
            transforms.append(NoTransform())

        return TransformationPipeline(
            pre_transforms=tuple(pre_transforms),
            transforms=tuple(transforms),
            merges_batch_patch_dims=merges_batch_patch_dims
        )

    def create_dataloader(
        self,
        dataset: DistributedSegmentedDummyData,
        is_train: bool = True
    ) -> DataLoader:
        """Create a DataLoader for the given dataset"""
        if self.dist_env['uses_dist']:
            sampler = DistributedSampler(
                dataset,
                shuffle=is_train,
                seed=self.base_seed,
                drop_last=self.config.dataset.drop_last_in_sampler
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            shuffle=(sampler is None and is_train),
            drop_last=self.config.dataset.drop_last_in_loader,
            pin_memory=self.config.dataset.pin_memory,
            prefetch_factor=self.config.dataset.prefetch_factor
        )

    def create_datasets(self):
        """Create training and evaluation datasets with dummy data"""
        transform_pipeline = self.setup_transforms()

        # Create dummy dataset configuration
        dummy_config = DummyDataConfig(
            C=self.config.dataset.num_channels,
            H=self.config.dataset.H,
            W=self.config.dataset.W,
            seg_size=self.config.dataset.seg_size,
            total_size=self.config.dataset.total_size,
            dist_rank=self.dist_env.get('rank', 0),
            dist_world_size=self.dist_env.get('world_size', 1),
            transforms=transform_pipeline.pre_transforms,
            dtype=torch.float32
        )

        # Create datasets using the dummy implementation
        train_dataset = DistributedSegmentedDummyData(dummy_config)
        eval_train_dataset = DistributedSegmentedDummyData(dummy_config)
        eval_val_dataset = DistributedSegmentedDummyData(dummy_config)

        return train_dataset, eval_train_dataset, eval_val_dataset

    def get_runtime_transforms(self) -> Tuple:
        """Get the runtime transforms for use during training"""
        return self.setup_transforms().transforms
