import pytest
from unittest.mock import Mock, patch, MagicMock
import yaml
import os
from typing import Tuple
from pathlib import Path

import pandas as pd
import numpy as np

import torch
import torch.distributed as dist
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
)

from maxie.config_training import TrainingConfig
from maxie.datasets.data_loading import TransformationPipeline, DatasetManager

def get_test_config_path():
    return Path(__file__).parent / "test_config_training.yaml"

def load_test_config():
    config_path = get_test_config_path()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def training_config():
    yaml_config = load_test_config()
    return TrainingConfig(**yaml_config)

@pytest.fixture
def dist_env():
    return {
        "uses_dist": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1
    }

@pytest.fixture
def mock_parquet():
    """Mock parquet file reading with complete metadata for 1000 zarr files with varying sizes"""
    with patch('pyarrow.parquet.read_table') as mock_read:
        num_files = 1000
        # Generate random sizes between 8 and 12 for some variety
        sizes = np.random.randint(8, 13, size=num_files)

        df = pd.DataFrame({
            'path': [f'test{i}.zarr' for i in range(num_files)],
            'start_idx': [0] * num_files,
            'end_idx': sizes.tolist(),
            'shape': [f'({size}, 1, 1920, 1920)' for size in sizes]
        })
        mock_read.return_value.to_pandas.return_value = df
        yield mock_read

@pytest.fixture
def mock_dist():
    """Mock torch.distributed functionality"""
    with patch('torch.distributed.init_process_group') as mock_init_pg, \
         patch('torch.distributed.is_initialized') as mock_is_init, \
         patch('torch.distributed.get_world_size') as mock_world_size, \
         patch('torch.distributed.get_rank') as mock_get_rank:
            # Configure the mocks
            mock_is_init.return_value = True
            mock_world_size.return_value = 1
            mock_get_rank.return_value = 0
            mock_init_pg.return_value = None

            yield {
                'init_process_group': mock_init_pg,
                'is_initialized': mock_is_init,
                'get_world_size': mock_world_size,
                'get_rank': mock_get_rank
            }

@pytest.fixture
def mock_zarr():
    """Mock zarr file access"""
    with patch('zarr.open') as mock_open:
        # Create mock zarr array with expected shape
        mock_array = MagicMock()
        mock_array.shape = (10, 1, 1920, 1920)
        # Create mock group that returns our array
        mock_group = MagicMock()
        mock_group.__getitem__.return_value = mock_array
        mock_open.return_value = mock_group
        yield mock_open

@pytest.fixture
def mock_dataset():
    """Create a mock dataset with required attributes"""
    mock = MagicMock(spec=DistributedZarrDataset)
    mock.start_idx = 0
    mock.end_idx = 240  # From seg_size in config
    mock.total_size = 1000
    mock.num_seg = 4
    mock.__len__.return_value = 10  # Important for DataLoader
    return mock

def test_transformation_pipeline():
    pipeline = TransformationPipeline()
    assert pipeline.pre_transforms == ()
    assert pipeline.transforms == ()
    assert pipeline.merges_batch_patch_dims == False

    custom_transforms = (NoTransform(),)
    pipeline = TransformationPipeline(
        pre_transforms=custom_transforms,
        transforms=custom_transforms,
        merges_batch_patch_dims=True
    )
    assert pipeline.pre_transforms == custom_transforms
    assert pipeline.transforms == custom_transforms
    assert pipeline.merges_batch_patch_dims == True

def test_setup_transforms(training_config, dist_env):
    manager = DatasetManager(training_config, dist_env)
    pipeline = manager.setup_transforms()

    assert isinstance(pipeline, TransformationPipeline)

    transform_set = training_config.dataset.transforms.set
    pre_transforms = pipeline.pre_transforms

    if transform_set.pad:
        assert any(isinstance(t, Pad) for t in pre_transforms)
    if transform_set.polar_center_crop:
        assert any(isinstance(t, PolarCenterCrop) for t in pre_transforms)
        assert any(isinstance(t, MergeBatchPatchDims) for t in pre_transforms)
    if not (transform_set.pad or transform_set.polar_center_crop):
        assert len(pre_transforms) == 1
        assert isinstance(pre_transforms[0], NoTransform)

@patch('maxie.datasets.data_loading.DistributedZarrDataset')
def test_create_datasets(mock_dataset_class, mock_parquet, mock_zarr, training_config, dist_env):
    # Configure mock dataset instance
    mock_instance = MagicMock(spec=DistributedZarrDataset)
    mock_instance.__len__.return_value = 10
    mock_dataset_class.return_value = mock_instance

    manager = DatasetManager(training_config, dist_env)
    train_dataset, eval_train_dataset, eval_val_dataset = manager.create_datasets()

    # Verify datasets were created with correct parameters
    assert mock_dataset_class.call_count == 3

    # Debug: print actual call arguments
    calls = mock_dataset_class.call_args_list
    for i, call in enumerate(calls):
        print(f"Call {i} args: {call.args}")
        print(f"Call {i} kwargs: {call.kwargs}")

    # Check positional arguments if path is passed as a positional arg
    if calls and calls[0].args:
        assert calls[0].args[0] == training_config.dataset.path_train
        assert calls[1].args[0] == training_config.dataset.path_train
        assert calls[2].args[0] == training_config.dataset.path_eval
    else:
        # If using kwargs
        assert calls[0].kwargs.get('parquet_file') == training_config.dataset.path_train  # Try 'parquet_file' instead of 'path'
        assert calls[1].kwargs.get('parquet_file') == training_config.dataset.path_train
        assert calls[2].kwargs.get('parquet_file') == training_config.dataset.path_eval

def test_create_dataloader(training_config, dist_env, mock_dataset, mock_dist):
    manager = DatasetManager(training_config, dist_env)

    # Test non-distributed setting
    dist_env['uses_dist'] = False
    loader = manager.create_dataloader(mock_dataset, is_train=True)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == training_config.dataset.batch_size
    assert loader.num_workers == training_config.dataset.num_workers
    assert loader.pin_memory == training_config.dataset.pin_memory
    assert loader.prefetch_factor == training_config.dataset.prefetch_factor

    # Test distributed setting
    dist_env['uses_dist'] = True
    loader = manager.create_dataloader(mock_dataset, is_train=True)
    assert isinstance(loader.sampler, DistributedSampler)

def test_get_runtime_transforms(training_config, dist_env):
    manager = DatasetManager(training_config, dist_env)
    transforms = manager.get_runtime_transforms()

    assert isinstance(transforms, tuple)

    transform_set = training_config.dataset.transforms.set
    transform_types = {
        'random_rotate': RandomRotate,
        'random_shift': RandomShift,
        'random_patch': RandomPatch,
    }

    for flag_name, transform_type in transform_types.items():
        if getattr(transform_set, flag_name):
            assert any(isinstance(t, transform_type) for t in transforms)
