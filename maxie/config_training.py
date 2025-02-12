import os
import yaml
import argparse
import logging

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)

class TransformSet(BaseModel):
    pad: bool
    uses_instance_norm: bool
    random_patch: bool
    random_rotate: bool
    random_shift: bool
    polar_center_crop: bool
    batch_sampler: bool

class TransformConfig(BaseModel):
    num_patch: int
    size_patch: int
    frac_shift_max: float
    angle_max: float
    var_size_patch: float
    sampling_fraction: Optional[float] = None
    H_pad: int
    W_pad: int
    Hv: int
    Wv: int
    sigma: float
    num_crop: int
    set: TransformSet

class DatasetConfig(BaseModel):
    path_train: str
    path_eval: str
    drop_last_in_sampler: bool
    drop_last_in_loader: bool
    batch_size: int
    num_workers: int
    seg_size: int
    pin_memory: bool
    prefetch_factor: int
    transforms: TransformConfig

class CheckpointConfig(BaseModel):
    directory: str
    prefix: Optional[str]
    path_chkpt_prev: Optional[str]
    chkpt_saving_iterations: Optional[int]
    preempt_metadata_path: Optional[str] = Field(default_factory=lambda: None)
    preempt_chkpt_saving_iterations: Optional[int]
    state_dict_type: str

    @validator('state_dict_type')
    def validate_state_dict_type(cls, v):
        if v not in ['full', 'sharded']:
            raise ValueError("state_dict_type must be either 'full' or 'sharded'")
        return v

class ModelConfig(BaseModel):
    from_scratch: bool
    hf_config: Dict[str, Any]

class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    fused: float
    grad_clip: float

    @validator('lr', 'weight_decay', 'beta1', 'beta2', 'grad_clip')
    def validate_positive(cls, v, field):
        if v < 0:
            raise ValueError(f"{field.name} must be positive")
        return v

class LRSchedulerConfig(BaseModel):
    warmup_iterations: int
    total_iterations: int
    min_lr: float
    scheduler_update_iterations: int

class ShardingStage(str, Enum):
    zero3 = "zero3"
    zero2 = "zero2"
    zero0 = "zero0"

class DistConfig(BaseModel):
    backend: str
    uses_unique_world_seed: bool
    dtype: str

    @validator('dtype')
    def validate_dtype(cls, v):
        if v not in ['float32', 'float16', 'bfloat16']:
            raise ValueError("dtype must be one of: float32, float16, bfloat16")
        return v

class LoggingConfig(BaseModel):
    directory: str
    prefix: str
    level: str

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"logging level must be one of: {valid_levels}")
        return v.upper()

class MiscConfig(BaseModel):
    max_epochs: int
    max_eval_iter: int
    max_eval_retry: int
    compiles_model: bool
    data_dump_on: bool = False
    cpu_only: bool = False
    peak_flops_per_sec: float
    monitors_dynamics: bool
    sharding_stage: ShardingStage

class TrainingConfig(BaseModel):
    checkpoint: CheckpointConfig
    dataset: DatasetConfig
    model: ModelConfig
    loss: Dict[str, Any]
    optim: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    dist: DistConfig
    logging: LoggingConfig
    misc: MiscConfig


def load_config(yaml_file:str) -> 'TrainingConfig':
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")
    with open(yaml_file, 'r') as fh:
        config_dict = yaml.safe_load(fh)
    return TrainingConfig(**config_dict)
