import os
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from contextlib import nullcontext
import tqdm
import time

from .lr_scheduler import CosineLRScheduler
from .utils.monitor import ActivationMonitor, monitor_param_update_metrics
from .utils.misc import is_action_due
from .config_training import TrainingConfig
from .utils.checkpoint import Checkpoint
from .utils_fsdp import (
    FullStateDictCheckpoint,
    ShardedStateDictCheckpoint,
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Track training progress and best results"""
    global_step: int = 0  # Counts parameter updates (macro-steps)
    best_val_loss: float = float('inf')
    start_idx: Optional[int] = None  # Dataset segment tracking
    end_idx: Optional[int] = None    # Dataset segment tracking

class Trainer:
    def __init__(self,
                 config: TrainingConfig,
                 dist_env: Dict[str, Any],
                 model_wrapper: 'ModelWrapper',
                 dataset_manager: 'DatasetManager',
                 checkpointer: Union[Checkpoint, FullStateDictCheckpoint, ShardedStateDictCheckpoint],
                 timestamp: str):
        """
        Initialize trainer with configuration and components

        Args:
            config: Training configuration
            dist_env: Distributed environment settings
            model_wrapper: Wrapped model with FSDP/mixed precision settings
            dataset_manager: Dataset and data loading manager
            timestamp: Timestamp string for checkpoint naming
        """
        self.config = config
        self.dist_env = dist_env
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.dataset_manager = dataset_manager
        self.checkpointer = checkpointer
        self.timestamp = timestamp

        # Device
        self.device = dist_env.get('device')

        # Training state
        self.state = TrainingState()

        # Setup components
        self._setup_training()
        self._setup_monitoring()
        self._setup_checkpointing()

    def _setup_checkpointing(self):
        # Load checkpoint if path is provided
        path_chkpt_prev = self.config.checkpoint.path_chkpt_prev
        if path_chkpt_prev:
            self.load_checkpoint(path_chkpt_prev)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint from given path

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if isinstance(self.checkpointer, (FullStateDictCheckpoint, ShardedStateDictCheckpoint)):
            # Only handle post-FSDP loading now
            iter_state = {}
            self.checkpointer.post_fsdp_load(
                self.dist_env['rank'],
                self.model,
                self.optimizer,
                self.scheduler,
                iter_state,
                checkpoint_path
            )

            # Update training state
            self.state.global_step = iter_state.get("global_step", 0)
            self.state.best_val_loss = iter_state.get("loss_min", float('inf'))
            self.state.start_idx = iter_state.get("start_idx")
            self.state.end_idx = iter_state.get("end_idx")

            if self.dist_env['rank'] == 0:
                logger.info(
                    f"Loaded checkpoint from {checkpoint_path} - "
                    f"step: {self.state.global_step}, "
                    f"best_val_loss: {self.state.best_val_loss}"
                )

    def save_checkpoint(self, is_best: bool = False):
        """
        Save checkpoint

        Args:
            is_best: Whether this is the best model so far
        """
        if self.dist_env['rank'] != 0:
            return

        # Prepare state dict
        iter_state = {
            "global_step": self.state.global_step,
            "loss_min": self.state.best_val_loss,
            "start_idx": self.state.start_idx,
            "end_idx": self.state.end_idx,
        }

        # Generate checkpoint path
        suffix = "best" if is_best else f"step_{self.state.global_step}"
        checkpoint_name = f"{self.timestamp}.{suffix}"
        if self.config.checkpoint.prefix:
            checkpoint_name = f"{self.config.checkpoint.prefix}.{checkpoint_name}"

        checkpoint_path = os.path.join(
            self.config.checkpoint.directory,
            checkpoint_name
        )

        # Save checkpoint
        self.checkpointer.save(
            self.dist_env['rank'],
            self.model,
            self.optimizer,
            self.scheduler,
            iter_state,
            checkpoint_path
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Handle preemption checkpoint if configured
        preempt_path = self.config.checkpoint.preempt_metadata_path
        if (preempt_path and
            is_action_due(self.state.global_step,
                         self.config.checkpoint.preempt_chkpt_saving_iterations)):

            preempt_name = f"{self.timestamp}.preempt"
            if self.config.checkpoint.prefix:
                preempt_name = f"{self.config.checkpoint.prefix}.{preempt_name}"

            preempt_checkpoint_path = os.path.join(
                self.config.checkpoint.directory,
                preempt_name
            )

            # Save preemption checkpoint
            self.checkpointer.save(
                self.dist_env['rank'],
                self.model,
                self.optimizer,
                self.scheduler,
                iter_state,
                preempt_checkpoint_path
            )
            logger.info(f"Saved preemption checkpoint to {preempt_checkpoint_path}")

            # Save metadata
            with open(preempt_path, "w") as f:
                f.write(preempt_checkpoint_path)
            logger.info(f"Saved preemption metadata to {preempt_path}")

    @torch.no_grad()
    def estimate_loss(
        self,
        dataloader,
        desc: str = '',
        dummy_input_shape: Optional[Tuple[int, ...]] = None
    ) -> float:
        """
        Estimate loss on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            desc: Description for progress bar
            dummy_input_shape: Shape for creating dummy input if needed for None batches

        Returns:
            Mean loss value
        """
        if self.dist_env['rank'] == 0:
            logger.debug(f"[RANK {self.dist_env['rank']}] - EVAL Entering")
        self.model.eval()

        # Set default number of iterations
        max_iter = self.config.misc.max_eval_iter or len(dataloader)

        # Initialize tracking tensors
        losses = torch.zeros(len(dataloader), device=self.device)
        num_samples = torch.zeros(len(dataloader), device=self.device)
        proc_masks = torch.zeros(len(dataloader), device=self.device)  # Process tracking mask
        none_mask = torch.zeros(len(dataloader), device=self.device)   # None batch mask

        # Evaluation loop
        for enum_idx, batch_data in enumerate(tqdm.tqdm(
            dataloader, 
            total=max_iter,
            desc=f'[RANK {self.dist_env["rank"]}] Eval{desc}'
        )):
            if enum_idx >= max_iter:
                break

            if self.dist_env['rank'] == 0:
                logger.debug(f"[RANK {self.dist_env['rank']}] EVAL - Pre fetching mini_batch {enum_idx}")

            # Handle None batch
            if batch_data is None:
                logger.debug(f"[RANK {self.dist_env['rank']}] Found None batch at idx {enum_idx}. Creating dummy input!")
                if dummy_input_shape is None:
                    raise ValueError("dummy_input_shape must be provided when handling None batches")
                batch_data = torch.zeros(
                    dummy_input_shape, 
                    dtype=self.model_wrapper.mixed_precision.param_dtype if self.model_wrapper.mixed_precision else torch.float32
                )
                none_mask[enum_idx] = 1

            # Process batch
            batch_input = batch_data.to(
                self.device,
                non_blocking=True,
                dtype=self.model_wrapper.mixed_precision.param_dtype if self.model_wrapper.mixed_precision else torch.float32
            )

            # Apply transforms
            transforms = self.dataset_manager.get_runtime_transforms()
            if transforms:
                for transform in transforms:
                    batch_input = transform(batch_input)

            if self.dist_env['rank'] == 0:
                logger.debug(f"[RANK {self.dist_env['rank']}] EVAL - Post fetching")

            # Forward pass
            with self.model_wrapper.autocast_context:
                if self.dist_env['rank'] == 0:
                    logger.debug(f"[RANK {self.dist_env['rank']}] EVAL - Forwarding")
                outputs = self.model(batch_input)

                if self.dist_env['rank'] == 0:
                    logger.debug(f"[RANK {self.dist_env['rank']}] EVAL - Loss")
                loss = outputs.loss

            # Save results
            losses[enum_idx] = loss
            num_samples[enum_idx] = len(batch_input)
            proc_masks[enum_idx] = 1

        # Handle nan values
        non_nan_mask = ~torch.isnan(losses)

        # Create final mask for valid values
        masks = torch.logical_and(proc_masks > 0, non_nan_mask)
        masks = torch.logical_and(masks, none_mask == 0)  # Exclude None batches

        # Calculate mean loss
        local_valid_losses = losses[masks].to(torch.float32)
        local_losses_mean = local_valid_losses.mean()

        # Handle nan across ranks
        world_nan_counter = torch.tensor(0, dtype=torch.int, device=self.device)
        local_nan_masks = torch.isnan(local_losses_mean)
        if local_nan_masks.any().item():
            logger.error(f"[RANK {self.dist_env['rank']}] EVAL ERROR: NaN encountered!")
            world_nan_counter += 1
            local_losses_mean = 0.0
        if self.dist_env['uses_dist']:
            dist.all_reduce(world_nan_counter, op=dist.ReduceOp.SUM)

        # Scale local loss
        local_losses_mean /= (self.dist_env['world_size'] - world_nan_counter + 1e-6)

        # Calculate final mean loss
        world_losses_mean = torch.zeros_like(local_losses_mean, dtype=torch.float32, device=self.device)
        world_losses_mean += local_losses_mean.to(torch.float32)
        if self.dist_env['uses_dist']:
            dist.all_reduce(world_losses_mean, op=dist.ReduceOp.SUM)

        # Optional data dumping
        if self.dist_env['rank'] == 0 and self.config.misc.data_dump_on:
            dump_data = {
                "losses": losses,
                "proc_masks": proc_masks,
                "non_nan_mask": non_nan_mask,
                "masks": masks,
                "local_valid_losses": local_valid_losses,
                "local_losses_mean": local_losses_mean,
                "world_losses_mean": world_losses_mean,
            }
            dump_dir = "data_dump"
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(
                dump_dir,
                f'{self.config.logging.prefix}.step{self.state.global_step}.eval.pt'
            )
            torch.save(dump_data, dump_path)

        self.model.train()
        return world_losses_mean.item()

    def _setup_training(self):
        """Initialize optimizer, scheduler and related components"""
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            fused=self.config.optim.fused
        )

        # Learning rate scheduler
        self.scheduler = CosineLRScheduler(
            optimizer=self.optimizer,
            warmup_iterations=self.config.lr_scheduler.warmup_iterations,
            total_iterations=self.config.lr_scheduler.total_iterations,
            min_lr=self.config.lr_scheduler.min_lr
        )

        # Gradient sync context
        self.grad_sync_context = (
            lambda enables_sync: nullcontext() if enables_sync or not self.dist_env['uses_dist'] 
            else self.model.no_sync()
        )

    def _setup_monitoring(self):
        """Setup optional training dynamics monitoring"""
        if self.config.misc.monitors_dynamics:
            self.act_monitor = ActivationMonitor(
                self.model,
                modules_to_monitor=(ACT2CLS[self.model.config.hidden_act],)
            )
            self.act_monitor.add_hooks()
        else:
            self.act_monitor = None


    def train(self):
        """Main training loop"""
        try:
            # Create datasets and set initial state if resuming
            train_dataset, eval_train_dataset, eval_val_dataset = self.dataset_manager.create_datasets()

            if self.state.start_idx is not None:
                train_dataset.start_idx = self.state.start_idx
                train_dataset.end_idx = self.state.end_idx

            # Create dataloaders
            train_loader = self.dataset_manager.create_dataloader(train_dataset)
            eval_train_loader = self.dataset_manager.create_dataloader(eval_train_dataset, is_train=False)
            eval_val_loader = self.dataset_manager.create_dataloader(eval_val_dataset, is_train=False)

            # Training loop
            self.model.train()
            grad_accum_steps = max(int(self.config.loss.get("grad_accum_steps")), 1)

            # Initialize accumulators
            accumulated_loss = torch.zeros(1, device=self.device)
            micro_step = 0
            t_start = time.monotonic()

            # Infinite training loop with step-based stopping
            while self.state.global_step < self.config.lr_scheduler.total_iterations:
                for batch_data in train_loader:
                    if batch_data is None:
                        continue

                    # Process batch
                    batch_input = batch_data.to(
                        self.device,
                        non_blocking=True,
                        dtype=self.model_wrapper.mixed_precision.param_dtype if self.model_wrapper.mixed_precision else torch.float32
                    )

                    # Apply transforms
                    transforms = self.dataset_manager.get_runtime_transforms()
                    if transforms:
                        for transform in transforms:
                            batch_input = transform(batch_input)

                    # Determine if gradient sync is needed
                    is_sync_step = (micro_step + 1) % grad_accum_steps == 0

                    # Forward and backward passes
                    with self.grad_sync_context(is_sync_step):
                        with self.model_wrapper.autocast_context:
                            outputs = self.model(batch_input)
                            loss = outputs.loss / grad_accum_steps

                        # Backward pass with gradient scaling
                        self.model_wrapper.scaler.scale(loss).backward()

                        # Accumulate loss for logging
                        accumulated_loss += loss.detach()

                    micro_step += 1

                    # Update parameters if gradient accumulation is complete
                    if is_sync_step:
                        # Gradient clipping
                        if self.config.optim.grad_clip > 0:
                            self.model_wrapper.scaler.unscale_(self.optimizer)
                            if (not self.dist_env['uses_dist']) or self.config.misc.sharding_stage == "zero0":
                                grad_norm = nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.optim.grad_clip
                                )
                            else:
                                grad_norm = self.model.clip_grad_norm_(self.config.optim.grad_clip)

                        # Parameter update
                        self.model_wrapper.scaler.step(self.optimizer)
                        self.model_wrapper.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                        # Update learning rate
                        if is_action_due(self.state.global_step, self.config.lr_scheduler.scheduler_update_iterations):
                            self.scheduler.step()

                        # Log training progress
                        if self.dist_env['rank'] == 0:
                            t_end = time.monotonic()
                            steps_per_sec = grad_accum_steps / (t_end - t_start)

                            logger.info(
                                f"step={self.state.global_step} | "
                                f"loss={accumulated_loss.item():.4f} | "
                                f"lr={self.scheduler.get_lr()[0]:.2e} | "
                                f"steps/sec={steps_per_sec:.2f}"
                            )

                        # Regular checkpoint saving
                        if is_action_due(self.state.global_step, self.config.checkpoint.chkpt_saving_iterations):
                            train_loss = self.estimate_loss(eval_train_loader)
                            val_loss = self.estimate_loss(eval_val_loader)

                            if self.dist_env['rank'] == 0:
                                logger.info(
                                    f"step={self.state.global_step} | "
                                    f"train_loss={train_loss:.4f} | "
                                    f"val_loss={val_loss:.4f}"
                                )

                            # Save best checkpoint if validation loss improved
                            if val_loss < self.state.best_val_loss:
                                self.state.best_val_loss = val_loss
                                self.save_checkpoint(is_best=True)

                            # Save regular checkpoint
                            self.save_checkpoint(is_best=False)

                        # Reset accumulators
                        accumulated_loss.zero_()
                        t_start = time.monotonic()

                        # Update training state
                        self.state.global_step += 1
                        self.state.start_idx = train_dataset.start_idx
                        self.state.end_idx = train_dataset.end_idx

                        # Check if training should stop
                        if self.state.global_step >= self.config.lr_scheduler.total_iterations:
                            break

                # Reset dataset if needed
                train_dataset.reset()

        except KeyboardInterrupt:
            logger.error(f"[RANK {self.dist_env['rank']}] Training was interrupted!")
            # Save checkpoint on interruption
            if self.dist_env['rank'] == 0:
                self.save_checkpoint(is_best=False)
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.act_monitor:
            self.act_monitor.remove_hooks()

        if self.dist_env['uses_dist']:
            dist.destroy_process_group()
