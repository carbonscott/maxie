import torch

def wrap_with_torch_dataloader(
    dataset,
    base_seed,
    drop_last_in_sampler,
    drop_last_in_loader,
    uses_dist,
    batch_size,
    num_workers,
    custom_collate,
    pin_memory,
    prefetch_factor,
    epoch,
    is_eval=False,
):
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=True,
        seed=base_seed,
        drop_last=drop_last_in_sampler
    ) if uses_dist else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False if is_eval else None,
        collate_fn=custom_collate,
        drop_last=drop_last_in_loader,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    return dataloader, sampler
