import os
import torch
import tqdm
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def estimate_loss(
    dataloader,
    model,
    autocast_context,
    max_iter = None,
    desc = '',
    device = 'cpu',
    dummy_input_shape = None,
    mixed_precision_dtype = torch.float32,
    transforms = None,
    data_dump_on = False,
    **kwargs
):
    ''' Estimate loss.
        The dataloader should be wrapped with Dataloader class or
        DistributedSampler class, best with shuffle being true.  The shuffle
        takes place before batching.
    '''
    # -- Setup
    uses_dist       = kwargs.get('uses_dist')
    dist_rank       = kwargs.get('dist_rank')
    dist_world_size = kwargs.get('dist_world_size')

    if dist_rank == 0:
        logger.debug(f"[RANK {dist_rank}] - EVAL Entering")
    model.eval()

    # !!!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok = True)

        fl_log_prefix = kwargs.get('fl_log_prefix')
        epoch         = kwargs.get('epoch')
        seg           = kwargs.get('seg')

    # -- Eval iterations
    # Set default number of iterations
    if max_iter is None:
        max_iter = len(dataloader)

    losses      = torch.zeros(len(dataloader), device = device)
    num_samples = torch.zeros(len(dataloader), device = device)
    proc_masks  = torch.zeros(len(dataloader), device = device)  # A mask to track the process
    none_mask   = torch.zeros(len(dataloader), device = device)  # Mask for None batches
    for enum_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = max_iter, desc = f'[RANK {dist_rank}] Eval{desc}'):    # (B, C, H, W)
        # Sample at most max_iter batches
        if enum_idx >= max_iter: break

        if dist_rank == 0:
            logger.debug(f"[RANK {dist_rank}] EVAL - Pre fetching mini_batch {enum_idx}")

        # Create dummy data for a None batch
        # FIXME: Better data cleaning will eliminate None batch
        if batch_data is None:
            logger.debug(f"[RANK {dist_rank}] Found None batch at batch idx {enum_idx}.  Creating a dummy input!!!")
            batch_data = torch.zeros(dummy_input_shape, dtype = mixed_precision_dtype)
            none_mask[enum_idx] = 1

        batch_input = batch_data
        batch_input = batch_input.to(device, non_blocking = True, dtype = mixed_precision_dtype)

        # Optional transform
        if transforms is not None:
            for enum_idx, trans in enumerate(transforms):
                batch_input = trans(batch_input)

        if dist_rank == 0:
            logger.debug(f"[RANK {dist_rank}] EVAL - Post fetching")

        with autocast_context:
            if dist_rank == 0:
                logger.debug(f"[RANK {dist_rank}] EVAL - Forwarding")
            batch_output = model(batch_input)

            if dist_rank == 0:
                logger.debug(f"[RANK {dist_rank}] EVAL - Loss")
            loss = batch_output.loss

        # !!!!!!!!!!!!!!!
        # !! Data dump !!
        # !!!!!!!!!!!!!!!
        if dist_rank == 0 and data_dump_on:
            mini_batch = enum_idx

            data_dump = {
                "batch_data"   : batch_data,
                "batch_output" : batch_output,
                "loss"         : loss,
            }
            path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}_minib{mini_batch}.loop.pt')
            torch.save(data_dump, path_data_dump)

        losses     [enum_idx] = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks [enum_idx] = 1

    # -- Handle nan
    # Obtain the nan mask
    non_nan_mask = ~torch.isnan(losses)

    # Get the actual mask of values that are from the processing loop and non nan
    masks = torch.logical_and(proc_masks>0, non_nan_mask)
    masks = torch.logical_and(masks, none_mask==0)  # Keep not-None elements

    # -- Mean loss over eval iterations
    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean  = local_valid_losses.mean()  # torch.isnan(torch.tensor([]).mean()) -> True

    # -- Mean loss over ranks
    # Survey the occurence of nan across ranks
    world_nan_counter = torch.tensor(0, dtype = torch.int, device = device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        logger.error(f"[RANK {dist_rank}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean  = 0.0    # Contribute to nothing in the reduced sum
    if uses_dist: dist.all_reduce(world_nan_counter, op = dist.ReduceOp.SUM)

    # Scale the local loss for the final reduced sum
    local_losses_mean /= (dist_world_size - world_nan_counter + 1e-6)

    # Calculate reduced sum as the final mean loss
    world_losses_mean  = torch.zeros_like(local_losses_mean, dtype = torch.float32, device = device)
    world_losses_mean += local_losses_mean.to(torch.float32)
    if uses_dist: dist.all_reduce(world_losses_mean, op = dist.ReduceOp.SUM)

    # !!!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        data_dump = {
            "losses"            : losses,
            "proc_masks"        : proc_masks,
            "non_nan_mask"      : non_nan_mask,
            "masks"             : masks,
            "local_valid_losses": local_valid_losses,
            "local_losses_mean" : local_losses_mean,
            "world_losses_mean" : world_losses_mean,
        }
        path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}.end.pt')
        torch.save(data_dump, path_data_dump)

    model.train()

    return world_losses_mean
