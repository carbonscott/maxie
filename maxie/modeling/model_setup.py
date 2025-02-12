from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import logging
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from ..utils_fsdp import (
    MemoryMaximizer,
    verify_bfloat_support,
    FullStateDictCheckpoint,
    ShardedStateDictCheckpoint,
)
from ..utils.checkpoint import Checkpoint
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEPreTrainedModel,
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAELayer,
)
from functools import partial
from contextlib import nullcontext

logger = logging.getLogger(__name__)

@dataclass
class ModelWrapper:
    model: nn.Module
    mixed_precision: Optional[MixedPrecision] = None
    autocast_context: Any = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None

class ModelBuilder:
    """Handles model initialization and FSDP setup"""

    def __init__(self, config: 'TrainingConfig', dist_env: Dict[str, Any]):
        self.config = config
        self.dist_env = dist_env
        self.device = f'cuda:{dist_env["local_rank"]}' if not config.misc.cpu_only and torch.cuda.is_available() else 'cpu'

        # Initialize checkpointer early for pre-FSDP loading
        checkpoint_func = {
            "full": FullStateDictCheckpoint,
            "sharded": ShardedStateDictCheckpoint,
        }[config.checkpoint.state_dict_type] if dist_env['uses_dist'] else Checkpoint
        self.checkpointer = checkpoint_func()

    def _patch_model_weights(self):
        """Patch model weight initialization methods"""
        def _init_weights_in_encoder(self, module):
            """Initialize encoder weights with scaled initialization"""
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Normalize init std by number of residual paths
                std = self.config.initializer_range
                std *= (2 * self.config.num_hidden_layers)**-0.5

                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        def _init_weights_in_decoder(self, module):
            """Initialize decoder weights with scaled initialization"""
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                std = self.config.initializer_range
                std *= (2 * self.config.decoder_num_hidden_layers)**-0.5

                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        # Apply patches
        ViTMAEModel._init_weights = _init_weights_in_encoder
        ViTMAEPreTrainedModel._init_weights = _init_weights_in_decoder

    def _setup_mixed_precision(self) -> tuple[Optional[MixedPrecision], Any, torch.cuda.amp.GradScaler]:
        """Configure mixed precision training settings"""
        # For float32, return None for mixed precision
        if self.config.dist.dtype == "float32":
            return None, nullcontext(), torch.cuda.amp.GradScaler(enabled=False)

        dtype_map = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }
        mixed_precision_dtype = dtype_map[self.config.dist.dtype]

        # Mixed precision config for FSDP
        mixed_precision = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        )

        # Autocast context
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        autocast_context = (
            torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)
            if device_type != 'cpu'
            else nullcontext()
        )

        # Gradient scaler for float16
        scaler_cls = ShardedGradScaler if self.dist_env['uses_dist'] else torch.cuda.amp.GradScaler
        scaler = scaler_cls(enabled=(self.config.dist.dtype == 'float16'))

        return mixed_precision, autocast_context, scaler

    def _setup_fsdp_config(self) -> Dict[str, Any]:
        """Configure FSDP settings"""
        # Sharding strategy
        sharding_map = {
            'zero3': ShardingStrategy.FULL_SHARD,
            'zero2': ShardingStrategy.SHARD_GRAD_OP,
            'zero0': ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_map[self.config.misc.sharding_stage]

        # Wrapping policy
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ViTMAELayer,
            },
        )

        # Activation checkpointing wrapper
        checkpoint_wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        return {
            'auto_wrap_policy': wrap_policy,
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'forward_prefetch': True,
            'sharding_strategy': sharding_strategy,
            'checkpoint_wrapper_fn': checkpoint_wrapper_fn,
        }

    def _apply_activation_checkpointing(self, model: nn.Module):
        """Apply activation checkpointing to transformer layers"""
        check_fn = lambda submodule: isinstance(submodule, ViTMAELayer)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=self._setup_fsdp_config()['checkpoint_wrapper_fn'],
            check_fn=check_fn
        )

    def build(self) -> ModelWrapper:
        """Build and configure the model"""
        # Initialize weights patching
        self._patch_model_weights()

        # Create model instance
        model_config = ViTMAEConfig(**self.config.model.hf_config)
        model = ViTMAEForPreTraining(model_config)

        # Handle pre-FSDP checkpoint loading if needed
        path_chkpt_prev = self.config.checkpoint.path_chkpt_prev
        if path_chkpt_prev and isinstance(self.checkpointer, (FullStateDictCheckpoint, ShardedStateDictCheckpoint)):
            if self.dist_env['rank'] == 0:
                logger.info(f"Loading pre-FSDP state from checkpoint: {path_chkpt_prev}")
            self.checkpointer.pre_fsdp_load(
                self.dist_env['rank'],
                model,
                path_chkpt_prev
            )

        # Log initial parameter stats if main process
        if self.dist_env['rank'] == 0:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    mean = module.weight.data.mean()
                    std = module.weight.data.std()
                    logger.info(f"logevent='INIT' | module={name} | mean={mean:.6f} | std={std:.6f}")

        # Setup mixed precision training
        mixed_precision, autocast_context, scaler = self._setup_mixed_precision()

        # Move model to device if not using FSDP
        if not self.dist_env['uses_dist']:
            model.to(self.device)
            return ModelWrapper(model, mixed_precision, autocast_context, scaler)

        # Configure FSDP
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        fsdp_config = self._setup_fsdp_config()

        model = FSDP(
            model,
            mixed_precision=mixed_precision,
            device_id=self.device,
            limit_all_gathers=True,
            use_orig_params=False,
            **fsdp_config
        )

        # Apply activation checkpointing
        self._apply_activation_checkpointing(model)

        # Log sharded parameter count
        if self.dist_env['rank'] == 0:
            param_count = sum(p.numel() for p in model.parameters())
            logger.debug(f"Sharded parameter count: {param_count*1e-6:.2f}M")

        return ModelWrapper(model, mixed_precision, autocast_context, scaler)

def setup_model(config: 'TrainingConfig', dist_env: Dict[str, Any]) -> ModelWrapper:
    """Main entry point for model setup"""
    builder = ModelBuilder(config, dist_env)
    return builder.build()
