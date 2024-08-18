import io
import csv
import json
import socket
import numpy as np
from bisect import bisect_right

from math import ceil
from itertools import islice

import torch
import torch.distributed as dist

from multiprocessing import shared_memory

from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..utils_fsdp import broadcast_dict
from ..perf import Timer

import logging
logger = logging.getLogger(__name__)

def _myrank():
    return dist.get_rank() if dist.is_initialized() else 0

class EventList:
    """Logical list of events, addressible by integers
       in 0, ..., len(EventList)-1
    """
    def __init__(self, path : str) -> None:
        logger.debug(f"[RANK %d] Loading json.", _myrank())
        with open(path, 'r') as f:
            self.entry_list = json.load(f)

        self.sizes = [entry['num_events'] \
                      if entry['events'] is None \
                      else len(entry['events']) \
                      for entry in self.entry_list]
        self.starts = np.array(self.sizes).cumsum()
        self.nitems = self.starts[-1]

    def __len__(self):
        return self.nitems

    def __getitem__(self, i):
        PSANA_ACCESS_MODE = 'idx'
        j = bisect_right(self.starts, i)
        i -= self.starts[j]
        entry = self.entry_list[j]
        if entry["events"] is not None:
            i = entry["events"][i]
        return entry["exp"], entry["run"], PSANA_ACCESS_MODE, \
               entry["detector_name"], i

# ----------------------------------------------------------------------- #
#  DATALOADER FOR TRAINING BY ALL RANKS
# ----------------------------------------------------------------------- #
@dataclass
class IPCDistributedSegmentedDatasetConfig:
    """Configuration for the Remote Distributed Segmented Dataset.

    Attributes:
        path_json (str)              : JSON file to configure the dataset list.
        seg_size (int)               : The segment size by each rank.
        world_size (int)             : Total number of distributed processes (ranks) in use.
        transforms (List)            : A list of transformations to apply to each data item.
        is_perf (bool)               : Flag to enable performance timing for transformations. Default is False.
        server_address (str)         : URL of the server to fetch data from. Defaults to 'http://localhost:5001'.
        loads_segment_in_init (bool) : Whether to load the first segment in the init.
        entry_per_cycle (int)        : Number of entries to go through in each experiment.
    """
    path_json             : str
    seg_size              : int
    world_size            : int
    transforms            : List
    is_perf               : bool  = False
    server_address        : Tuple = ('localhost', 5000)
    loads_segment_in_init : bool  = False
    entry_per_cycle       : int   = 1
    debug                 : bool  = False

class IPCDistributedSegmentedDataset(Dataset):
    """A dataset class designed for fetching and distributing segments of data
    in a distributed training environment.

    This class allows for efficient data loading and processing across multiple
    distributed processes.
    """
    def __init__(self, config: IPCDistributedSegmentedDatasetConfig):
        self.seg_size               = config.seg_size
        self.world_size             = config.world_size
        self.server_address         = config.server_address
        self.transforms             = config.transforms
        self.is_perf                = config.is_perf
        self.loads_segment_in_init  = config.loads_segment_in_init
        self.entry_per_cycle        = config.entry_per_cycle
        self.debug                  = config.debug

        self.events          = EventList(config.path_json)
        self.total_size      = len(self.events)

        self.json_entry_gen  = None
        self.current_dataset = None
        self.start_idx       = 0
        self.end_idx         = 0

        if self.loads_segment_in_init: self.set_start_idx(start_idx = 0)

    def reset(self):
        self.start_idx       = 0
        self.end_idx         = 0
        self.json_entry_gen  = None
        self.current_dataset = None
        if self.debug:
            logger.debug(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] Dataset reset done.")

    def _init_entry_generator(self):
        if self.debug:
            logger.debug("[RANK %d] Initializing entry generator.", _myrank())
        PSANA_ACCESS_MODE = 'idx'
        entry_gens = []
        for entry in self.events.entry_list:
            exp           = entry['exp'          ]
            run           = entry['run'          ]
            detector_name = entry['detector_name']
            events        = entry['events'       ]
            num_events    = entry['num_events'   ]
            if events is None:
                events = range(num_events)
            entry_gen = self._entry_generator(exp, run, PSANA_ACCESS_MODE, detector_name, events)
            entry_gens.append(entry_gen)

        return self._round_robin_generator(entry_gens, self.entry_per_cycle)

    def _entry_generator(self, exp, run, psana_access_mode, detector_name, events):
        for event in events:
            yield (exp, run, psana_access_mode, detector_name, event)

    def _round_robin_generator(self, entry_gens, entry_per_cycle):
        """
        Go through up to certain number of examples for each exp and
        then move on to the next exp. Then, repeat this cycle until all
        generators have been exhausted.
        """
        while len(entry_gens):
            for entry_gen in entry_gens:
                for _ in range(entry_per_cycle):
                    try:
                        yield next(entry_gen)
                    except StopIteration:
                        entry_gens.remove(entry_gen)
                        break

    def calculate_end_idx(self):
        """
        end_idx is not inclusive (up to, but not including end_idx)
        """
        # Calculate and return the end index for the current dataset segment.
        return min(self.start_idx + self.seg_size * self.world_size, self.total_size)

    @property
    def num_seg(self):
        return ceil(self.total_size / (self.seg_size * self.world_size))

    def update_dataset_segment(self):
        if self.debug:
            logger.debug(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] Updating segment to {self.start_idx}-{self.end_idx}.")
        # Trick islice to return a subset of events at a time
        return list(islice(self.json_entry_gen, 0, self.end_idx - self.start_idx))

    def set_start_idx(self, start_idx):
        # Reset if the start_idx points to the end of the dataset
        if start_idx == self.total_size:
            self.reset()

        if self.debug:
            logger.debug(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] Setting start idx to {start_idx}.")
        self.start_idx = start_idx
        self.end_idx   = self.calculate_end_idx()

        # Optionally reset and/or advance the generator
        if self.json_entry_gen is None:
            # Initialize the generator for a resumption or rewind
            json_entry_gen = self._init_entry_generator()
            self.json_entry_gen = islice(json_entry_gen, self.start_idx, None)

        self.current_dataset = self.update_dataset_segment()

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # Ensure idx is within the bounds of the current segment
        if idx >= (self.end_idx - self.start_idx):
            raise IndexError("Index out of range for the current segment")

        # Obtain dataset handle
        exp, run, access_mode, detector_name, event = self.current_dataset[idx]

        # Fetch event
        image = fetch_event(self.server_address, exp, run, access_mode, detector_name, event)    # psana image: (H, W)

        # [DEBUG]
        if self.debug:
            logger.debug(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] exp={exp}, run={run}, detector_name={detector_name}, event={event}.")

        # Apply transforms
        image_tensor = None
        if image is not None and self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])  # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = None, is_on = self.is_perf):
                    image_tensor = trans(image_tensor, detector_name=detector_name)

        return image_tensor

    def save_checkpoint(self, checkpoint_path, rank):
        if rank == 0:
            checkpoint = {
                'end_idx' : self.end_idx,
                'seg_size': self.seg_size
            }
            torch.save(checkpoint, checkpoint_path)
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint_and_broadcast(self, checkpoint_path, rank, device):
        checkpoint = None
        if rank == 0:
            checkpoint = torch.load(checkpoint_path)
        checkpoint = broadcast_dict(checkpoint, src=0, device=device)

        if checkpoint:
            self.set_start_idx(checkpoint.get('end_idx', 0))
            if 'seg_size' in checkpoint and checkpoint['seg_size'] != self.seg_size:
                warnings.warn(f"seg_size has been changed from {checkpoint['seg_size']} to {self.seg_size}. Resetting to {checkpoint['seg_size']}.")
                self.seg_size = checkpoint['seg_size']

        if dist.is_initialized():
            dist.barrier()



# ----------------------------------------------------------------------- #
#  DATALOADER FOR EVALUATION BY RANK0
# ----------------------------------------------------------------------- #
@dataclass
class IPCDatasetConfig:
    """Configuration for the Inter-Processor Communication based Dataset.

    Attributes:
        path_json (str)              : JSON file to configure the dataset list.
        transforms (List): A list of transformations to apply to each data item.
        is_perf (bool): Flag to enable performance timing for transformations. Default is False.
        server_address (str): URL of the server to fetch data from. Defaults to 'http://localhost:5001'.
    """
    path_json                : str
    transforms               : List
    is_perf                  : bool = False
    server_address           : Tuple = ('localhost', 5000)

class IPCDataset(Dataset):
    """A dataset class designed for fetching data through Inter-Processor
    Communication.

    This class allows for efficient data loading and processing across multiple
    distributed processes.
    """
    def __init__(self, config: IPCDatasetConfig):
        self.events         = EventList(config.path_json)
        self.server_address = config.server_address
        self.transforms     = config.transforms
        self.is_perf        = config.is_perf

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        # Obtain dataset handle
        exp, run, access_mode, detector_name, event = self.events[idx]

        # Fetch event
        image = self.fetch_event(exp, run, access_mode, detector_name, event)    # psana image: (H, W)

        # Apply transforms
        image_tensor = None
        if image is not None and self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])    # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = None, is_on = self.is_perf):
                    image_tensor = trans(image_tensor)

        return image_tensor[0]    # Dataloader only wants data with shape of (C, H, W)

def fetch_event(server_address, exp, run, access_mode, detector_name, event):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)

        # Send request
        request_data = json.dumps({
            'exp'          : exp,
            'run'          : int(run),
            'access_mode'  : access_mode,
            'detector_name': detector_name,
            'event'        : int(event),
            'mode'         : 'image',
        })
        sock.sendall(request_data.encode('utf-8'))

        # Receive and process response
        response_data = sock.recv(4096).decode('utf-8')

        # Process response with a non-empty string
        result = None
        if len(response_data):
            # Load the reponse data
            response_json = json.loads(response_data)

            if 'error' in response_json:
                logger.debug(f"Server error: {response_json['error']}")
                if response_json['traceback'] is not None: logger.debug(response_json['traceback'])
            else:
                # Use the JSON data to access the shared memory
                shm_name = response_json['name']
                shape    = response_json['shape']
                dtype    = np.dtype(response_json['dtype'])

                # Initialize shared memory outside of try block to ensure it's in scope for finally block
                shm = None
                try:
                    # Access the shared memory
                    shm = shared_memory.SharedMemory(name=shm_name)
                    data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                    # Convert to numpy array (this creates a copy of the data)
                    result = np.array(data_array)
                finally:
                    # Ensure shared memory is closed even if an exception occurs
                    if shm:
                        shm.close()
                        shm.unlink()

            # Send acknowledgment after successfully accessing shared memory
            sock.sendall("ACK".encode('utf-8'))

        return result
