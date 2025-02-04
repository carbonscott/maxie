from typing import Tuple, Optional, List
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader, DistributedSampler

from maxie.datasets.zarr_dataset import DistributedZarrDataset
from maxie.tensor_transforms import (
    NoTransform,
    PolarCenterCrop,
    MergeBatchPatchDims,
    Pad,
    InstanceNorm,
    RandomPatch,
    RandomRotate,
    RandomShift,
    Patchify,
    Norm,
    BatchSampler,
)

from maxie.config_training import TrainingConfig

class TransformationPipeline(BaseModel):
    """Container for pre-transforms and runtime transforms"""
    pre_transforms: Tuple = ()
    transforms: Tuple = ()
    merges_batch_patch_dims: bool = False

class DatasetManager:
    """Manages dataset creation and transformations for training and evaluation"""

    def __init__(self, config: TrainingConfig, dist_env: dict):
        """
        Initialize DatasetManager with config and distributed environment settings

        Args:
            config: Training configuration using Pydantic models
            dist_env: Dictionary containing distributed training settings
        """
        self.config = config
        self.dist_env = dist_env
        self.base_seed = 0

    def setup_transforms(self) -> TransformationPipeline:
        """
        Set up all data transformations based on configuration

        Returns:
            TransformationPipeline containing pre-transforms and runtime transforms
        """
        transforms_config = self.config.dataset.transforms
        transform_set = transforms_config.set

        # Pre-transforms setup
        pre_transforms = []

        # Add padding if configured
        if transform_set.pad:
            pre_transforms.append(Pad(
                transforms_config.H_pad,
                transforms_config.W_pad
            ))

        # Add polar center crop if configured
        if transform_set.polar_center_crop:
            pre_transforms.append(PolarCenterCrop(
                Hv=transforms_config.Hv,
                Wv=transforms_config.Wv,
                sigma=transforms_config.sigma,
                num_crop=transforms_config.num_crop,
            ))

        # Add batch-patch dimension merging if using polar center crop
        merges_batch_patch_dims = transform_set.polar_center_crop
        if merges_batch_patch_dims:
            pre_transforms.append(MergeBatchPatchDims())

        # If no pre-transforms are configured, add NoTransform
        if not pre_transforms:
            pre_transforms.append(NoTransform())

        # Runtime transforms setup
        transforms = []

        # Add instance normalization if configured
        if transform_set.uses_instance_norm:
            transforms.append(InstanceNorm())

        # Add random rotations if configured
        if transform_set.random_rotate:
            transforms.append(RandomRotate(
                angle_max=transforms_config.angle_max
            ))

        # Add random shifts if configured
        if transform_set.random_shift:
            transforms.append(RandomShift(
                frac_y_shift_max=transforms_config.frac_shift_max,
                frac_x_shift_max=transforms_config.frac_shift_max,
            ))

        # Add random patch if configured
        if transform_set.random_patch:
            transforms.append(RandomPatch(
                num_patch=transforms_config.num_patch,
                size_patch=transforms_config.size_patch,
                var_size_patch=transforms_config.var_size_patch
            ))

        # If no runtime transforms are configured, add NoTransform
        if not transforms:
            transforms.append(NoTransform())

        return TransformationPipeline(
            pre_transforms=tuple(pre_transforms),
            transforms=tuple(transforms),
            merges_batch_patch_dims=merges_batch_patch_dims
        )

    def create_dataloader(
        self,
        dataset: DistributedZarrDataset,
        is_train: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader for the given dataset

        Args:
            dataset: The dataset to create a loader for
            is_train: Whether this is a training dataloader

        Returns:
            Configured DataLoader instance
        """
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
        """
        Create training and evaluation datasets with appropriate transforms

        Returns:
            Tuple of (train_dataset, eval_train_dataset, eval_val_dataset)
        """
        transform_pipeline = self.setup_transforms()

        # Create training dataset
        train_dataset = DistributedZarrDataset(
            self.config.dataset.path_train,
            seg_size=self.config.dataset.seg_size,
            transforms=transform_pipeline.pre_transforms,
            seed=self.base_seed
        )

        # Create evaluation dataset from training data
        eval_train_dataset = DistributedZarrDataset(
            self.config.dataset.path_train,
            seg_size=self.config.dataset.seg_size,
            transforms=transform_pipeline.pre_transforms,
            seed=self.base_seed
        )

        # Create evaluation dataset from validation data
        eval_val_dataset = DistributedZarrDataset(
            self.config.dataset.path_eval,
            seg_size=self.config.dataset.seg_size,
            transforms=transform_pipeline.pre_transforms,
            seed=self.base_seed
        )

        return train_dataset, eval_train_dataset, eval_val_dataset

    def get_runtime_transforms(self) -> Tuple:
        """
        Get the runtime transforms for use during training

        Returns:
            Tuple of transform objects
        """
        return self.setup_transforms().transforms
