import torch
from torch.utils.data import Dataset

import os
import torch.distributed as dist
from collections import deque
import zarr
import pyarrow.parquet as pq

import logging

logger = logging.getLogger(__name__)

class DistributedZarrDataset(Dataset):
    """
    A distributed dataset class for handling large datasets stored in Zarr format.

    Args:
        parquet_file (str)          : Path to the Parquet file containing Zarr file information.
        seg_size     (int)          : Size of each segment per rank.
        seed         (int, optional): Seed for random number generation. Defaults to 42.
        cache_size   (int, optional): Maximum number of Zarr files to keep in cache. Defaults to 100.
    """

    def __init__(self, parquet_file, seg_size, transforms = None, seed=42, cache_size=100):
        # Initialize ZarrDataset components
        self.metadata = pq.read_table(parquet_file).to_pandas()
        self.cumulative_sizes = torch.cumsum(torch.tensor([eval(shape)[0] for shape in self.metadata['shape']]), dim=0)
        self.total_images = self.cumulative_sizes[-1].item()

        # Initialize SegmentManager components
        self.world_size      = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank            = int(os.environ.get("RANK", "0"))
        self.seg_size        = seg_size
        self.global_seg_size = self.seg_size * self.world_size
        self.segment_manager = SegmentManager(self.total_images, self.seg_size)
        self.segment_manager.set_start_idx(0)

        # Shuffling
        self.generator = torch.Generator().manual_seed(seed)
        self.index_map = torch.arange(self.total_images)
        self.shuffle()

        # Zarr caching
        self.zarr_cache = {}
        self.cache_queue = deque(maxlen=cache_size)

        # Transform
        self.transforms = transforms

    def shuffle(self):
        """
        Perform a global shuffle of the dataset.
        """
        rand_idx = torch.randperm(self.total_images, generator=self.generator)
        self.index_map = self.index_map[rand_idx]
        logger.debug(f"Rank {self.rank}: Global shuffle performed")

    def __len__(self):
        """
        Returns the length of the current segment.
        """
        return self.segment_manager.end_idx - self.segment_manager.start_idx

    def __getitem__(self, idx):
        """
        Fetch an item from the dataset using global shuffle.

        Args:
            idx (int): Index of the item to fetch within the current segment.

        Returns:
            torch.Tensor: The image tensor.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        global_idx = self.segment_manager.start_idx + idx
        original_idx = self.index_map[global_idx].item()
        return self._fetch_image(original_idx)

    def _fetch_image(self, original_idx):
        """
        Fetch an image from the appropriate Zarr file.

        Args:
            original_idx (int): The original index of the image in the dataset.

        Returns:
            torch.Tensor: The image tensor.
        """
        file_idx, zarr_idx = self._get_file_and_zarr_indices(original_idx)
        zarr_path = self.metadata.iloc[file_idx]['absolute_path']

        if zarr_path not in self.zarr_cache:
            if len(self.cache_queue) == self.cache_queue.maxlen:  # Queue is full???
                oldest_path = self.cache_queue[0]
                self.zarr_cache[oldest_path].store.close()
                del self.zarr_cache[oldest_path]

            self.zarr_cache[zarr_path] = zarr.open(zarr_path, mode='r')
            self.cache_queue.append(zarr_path)  # deque auto-clean items in FIFO manner

        z = self.zarr_cache[zarr_path]
        image = z['data'][zarr_idx]

        if self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])  # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                image_tensor = trans(image_tensor)

        logger.debug(f"Rank {self.rank}: Loaded image from {zarr_path}, index {zarr_idx}")
        return image_tensor[0]  # (B=1,C,H,W) -> (C,H,W)

    def _get_file_and_zarr_indices(self, original_idx):
        """
        Get the file index and Zarr index for a given original index.

        Args:
            original_idx (int): The original index of the image in the dataset.

        Returns:
            tuple: (file_idx, zarr_idx)
        """
        file_idx = torch.searchsorted(self.cumulative_sizes, original_idx, right=True).item()
        zarr_idx = original_idx - self.cumulative_sizes[file_idx - 1].item() if file_idx > 0 else original_idx
        return file_idx, int(zarr_idx)

    def get_metadata(self, idx):
        """
        Get metadata for a specific index.

        Args:
            idx (int): Index of the item within the current segment.

        Returns:
            pandas.Series: Metadata for the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        global_idx = self.segment_manager.start_idx + idx
        original_idx = self.index_map[global_idx].item()
        file_idx, _ = self._get_file_and_zarr_indices(original_idx)
        return self.metadata.iloc[file_idx]

    def save_checkpoint(self, checkpoint_path):
        """
        Save the current state of the dataset to a checkpoint.

        Args:
            checkpoint_path (str): Path to save the checkpoint.
        """
        if self.rank == 0:
            checkpoint = {
                'end_idx'        : self.segment_manager.end_idx,
                'seg_size'       : self.segment_manager.seg_size,
                'index_map'      : self.index_map,
                'generator_state': self.generator.get_state()
            }
            torch.save(checkpoint, checkpoint_path)
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, checkpoint_path):
        """
        Load the state of the dataset from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if dist.is_initialized():
            if self.rank == 0:
                checkpoint      = torch.load(checkpoint_path)
                end_idx         = torch.tensor(checkpoint['end_idx'], dtype=torch.long)
                seg_size        = torch.tensor(checkpoint['seg_size'], dtype=torch.long)
                index_map       = checkpoint['index_map']
                generator_state = checkpoint['generator_state']
            else:
                end_idx         = torch.tensor(0, dtype=torch.long)
                seg_size        = torch.tensor(0, dtype=torch.long)
                index_map       = torch.zeros(self.total_images, dtype=torch.long)
                generator_state = torch.empty_like(self.generator.get_state())

            dist.broadcast(end_idx        , src=0)
            dist.broadcast(seg_size       , src=0)
            dist.broadcast(index_map      , src=0)
            dist.broadcast(generator_state, src=0)

            self.segment_manager.seg_size = seg_size.item()
            self.segment_manager.global_seg_size = self.segment_manager.seg_size * self.world_size
            self.segment_manager.set_start_idx(end_idx.item())
            self.index_map = index_map
            self.generator.set_state(generator_state)
        else:
            checkpoint = torch.load(checkpoint_path)
            self.segment_manager.seg_size = checkpoint['seg_size']
            self.segment_manager.global_seg_size = self.segment_manager.seg_size * self.world_size
            self.segment_manager.set_start_idx(checkpoint['end_idx'])
            self.index_map = checkpoint['index_map']
            self.generator.set_state(checkpoint['generator_state'])

    def close(self):
        """
        Close all open Zarr files and clear the cache.
        """
        for z in self.zarr_cache.values():
            z.store.close()
        self.zarr_cache.clear()
        self.cache_queue.clear()

class SegmentManager:
    """
    Manages data segmentation for distributed processing.

    Args:
        total_size (int): Total size of the dataset.
        seg_size   (int): Size of each segment per rank.
    """

    def __init__(self, total_size, seg_size):
        self.total_size      = total_size
        self.seg_size        = seg_size
        self.world_size      = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank            = int(os.environ.get("RANK", "0"))
        self.global_seg_size = self.seg_size * self.world_size
        self.start_idx       = 0
        self.end_idx         = 0

    def calculate_end_idx(self):
        """Calculate the end index for the current segment."""
        return min(self.start_idx + self.global_seg_size, self.total_size)

    def set_start_idx(self, start_idx):
        """
        Set the start index for the current segment.

        Args:
            start_idx (int): The new start index.
        """
        if start_idx >= self.total_size:
            self.reset()
        else:
            self.start_idx = start_idx
            self.end_idx = self.calculate_end_idx()

    def reset(self):
        """Reset the segment manager to the initial state."""
        self.start_idx = 0
        self.end_idx = 0
