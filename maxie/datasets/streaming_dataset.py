import torch
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
import io
import numpy as np
from pynng import Pull0, Timeout
import threading
import queue
import time
import ast
import logging
import signal
import os
import pathlib
import uuid
import socket

logger = logging.getLogger(__name__)

@dataclass
class StreamingDataConfig:
    C: int                                      # Number of channels
    H: int                                      # Height of tensor
    W: int                                      # Width of tensor
    address: Optional[str] = None               # Single address (legacy support)
    addresses: List[str] = None                 # List of addresses to pull data from
    address_assignment: str = "round-robin"     # How to assign addresses to nodes
    node_address_map: dict = None               # Explicit node->address mapping
    queue_size: int = 128                       # Size of shared queue
    timeout_ms: int = 1000                      # Socket timeout in milliseconds
    max_wait_time: int = 60                     # Max time to wait if queue is empty (seconds)
    connect_timeout: int = 10                   # Seconds to wait for initial connection
    transforms: Union[None, List, Tuple] = None # Data transforms
    dtype: torch.dtype = None                   # Optional dtype conversion
    dist_rank: int = 0                          # Distributed rank (for logging)
    dist_world_size: int = 1                    # World size (for stats only)
    local_rank: int = 0                         # Local rank (within node)
    num_nodes: int = 1                          # Number of nodes
    node_id: int = 0                            # ID of this node
    lock_dir: str = None                        # Directory for lock files

    def __post_init__(self):
        """Validate and normalize configuration after initialization"""
        # Handle legacy single address configuration
        if self.addresses is None:
            if self.address is not None:
                self.addresses = [self.address]
            else:
                raise ValueError("Either 'address' or 'addresses' must be provided")

        # Ensure node_address_map is initialized
        if self.node_address_map is None:
            self.node_address_map = {}

    def get_address_for_node(self, node_id):
        """Determine which address this node should connect to"""
        if not self.addresses:
            raise ValueError("No addresses configured")

        if len(self.addresses) == 1:
            # Only one address, all nodes use it
            return self.addresses[0]

        # Multiple addresses - use assignment strategy
        if self.address_assignment == "explicit":
            # Use explicit mapping if provided
            if node_id in self.node_address_map:
                return self.node_address_map[node_id]
            else:
                # Fall back to modulo if node_id not in map
                return self.addresses[node_id % len(self.addresses)]

        elif self.address_assignment == "random":
            # Use a seeded random assignment based on node_id
            # This ensures the same node always gets the same address
            import random
            random.seed(node_id)
            return random.choice(self.addresses)

        else:  # Default to "round-robin"
            # Simple round-robin assignment
            return self.addresses[node_id % len(self.addresses)]

class StreamingDataset(IterableDataset):
    """
    A streaming dataset for multi-node training that:
    - Has one socket connection per node (not per rank)
    - Waits instead of generating random tensors if data isn't available
    - Tracks global indices for simple checkpointing/resumption
    - Works with distributed training across multiple nodes
    """

    # Class-level variables to manage node-level connections
    _node_queues = {}          # Shared queues per node
    _node_pullers = {}         # Puller threads per node
    _node_stop_flags = {}      # Stop flags per node
    _node_stats = {}           # Statistics per node
    _node_locks = {}           # Locks to ensure only one connection per node
    _node_ref_counts = {}      # Reference counts for cleanup

    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        self.transforms = config.transforms
        self.rank = config.dist_rank
        self.world_size = config.dist_world_size
        self.local_rank = config.local_rank
        self.node_id = config.node_id

        # Determine which address this node should use
        self.node_address = config.get_address_for_node(self.node_id)

        # Create a unique key for this node that includes the address
        # This allows different nodes to connect to different addresses
        self.node_key = f"node_{self.node_id}_{self.node_address}"

        # Initialize reporting variables
        self.last_report_time = time.time()
        self.report_interval = 10.0  # seconds

        # Track highest global index seen by this rank
        self.highest_index = -1

        # Set up node-level resources if this is the first rank on this node
        self._setup_node_resources()

        # Register signal handlers for cleanup
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_node_resources(self):
        """Set up shared resources at the node level"""

        # Use a lock to ensure only one rank creates the resources
        if self.node_key not in StreamingDataset._node_locks:
            StreamingDataset._node_locks[self.node_key] = threading.Lock()

        with StreamingDataset._node_locks[self.node_key]:
            # Check if resources exist
            if self.node_key not in StreamingDataset._node_queues:
                # This is the first rank on this node - create shared resources
                logger.info(f"[RANK {self.rank}] Creating node-level resources for node {self.node_id}")

                # Create a shared queue for this node
                StreamingDataset._node_queues[self.node_key] = mp.Queue(self.config.queue_size)

                # Create a stop flag for the puller thread
                StreamingDataset._node_stop_flags[self.node_key] = mp.Event()

                # Create shared statistics counters
                StreamingDataset._node_stats[self.node_key] = {
                    'total_received': mp.Value('i', 0),
                    'total_yielded': mp.Value('i', 0),
                    'latest_index': mp.Value('i', -1),
                }

                # Initialize reference counter
                StreamingDataset._node_ref_counts[self.node_key] = 0

                # Start the puller thread for this node
                self._start_node_puller()

            # Increment the reference counter
            StreamingDataset._node_ref_counts[self.node_key] += 1

            # Log node resource status
            logger.info(f"[RANK {self.rank}] Using node-level resources for node {self.node_id}, "
                      f"ref count = {StreamingDataset._node_ref_counts[self.node_key]}")

    def _start_node_puller(self):
        """Start the puller thread for this node if it's not already running"""
        if self.node_key not in StreamingDataset._node_pullers or not StreamingDataset._node_pullers[self.node_key].is_alive():
            # Log that we're starting a thread
            logger.info(f"[RANK {self.rank}] Starting node-level puller thread for {self.config.address}")

            # Create and start the thread with the node-specific address
            StreamingDataset._node_pullers[self.node_key] = threading.Thread(
                target=self._pull_data_thread,
                args=(
                    self.node_address,  # Use the node-specific address
                    self.config.timeout_ms,
                    self.node_id,
                    self.rank,
                    StreamingDataset._node_queues[self.node_key],
                    StreamingDataset._node_stop_flags[self.node_key],
                    StreamingDataset._node_stats[self.node_key],
                )
            )
            StreamingDataset._node_pullers[self.node_key].daemon = True
            StreamingDataset._node_pullers[self.node_key].start()

            # Wait for some initial data
            got_data = False
            start_time = time.time()
            logger.info(f"[RANK {self.rank}] Waiting for initial data...")

            while time.time() - start_time < self.config.connect_timeout:
                try:
                    if not StreamingDataset._node_queues[self.node_key].empty():
                        got_data = True
                        break
                except Exception as e:
                    logger.warning(f"[RANK {self.rank}] Error checking queue: {e}")
                time.sleep(0.1)

            if got_data:
                logger.info(f"[RANK {self.rank}] Received initial data")
            else:
                logger.warning(f"[RANK {self.rank}] No initial data received after {self.config.connect_timeout}s")

    def _signal_handler(self, sig, frame):
        """Handle signals to clean up resources properly"""
        logger.info(f"[RANK {self.rank}] Received signal {sig}, cleaning up...")
        self.close()

        # Call the original handler
        if sig == signal.SIGINT and self._original_sigint_handler:
            self._original_sigint_handler(sig, frame)
        elif sig == signal.SIGTERM and self._original_sigterm_handler:
            self._original_sigterm_handler(sig, frame)

    @staticmethod
    def _pull_data_thread(address, timeout_ms, node_id, rank, node_queue, stop_flag, node_stats):
        """Background thread to continuously pull data and fill the queue"""
        # Make sure we have our own report time for the thread
        thread_report_time = time.time()

        try:
            with Pull0(dial=address) as sock:
                sock.recv_timeout = timeout_ms

                logger.info(f"[NODE {node_id}] Puller initialized, dialing to {address}")

                while not stop_flag.is_set():
                    try:
                        # Pull data
                        data = sock.recv()

                        # Track received data
                        with node_stats["total_received"].get_lock():
                            node_stats["total_received"].value += 1

                        # Extract tensor and metadata
                        tensor, metadata = StreamingDataset._parse_data(data, rank)

                        if tensor is not None:
                            # Put in the queue (block if full)
                            try:
                                node_queue.put((tensor, metadata), block=True, timeout=1.0)
                            except queue.Full:
                                logger.warning(f"[NODE {node_id}] Queue full, dropping tensor")
                                continue

                            # Update latest global index if available
                            if metadata and 'index' in metadata:
                                with node_stats["latest_index"].get_lock():
                                    node_stats["latest_index"].value = max(
                                        node_stats["latest_index"].value,
                                        metadata['index']
                                    )

                        # Periodic reporting (thread-local time)
                        current_time = time.time()
                        if current_time - thread_report_time > 10.0:  # 10 seconds
                            try:
                                queue_size = node_queue.qsize()
                            except:
                                queue_size = "Unknown"  # qsize() not reliable on all platforms

                            received = node_stats["total_received"].value
                            latest_idx = node_stats["latest_index"].value

                            logger.info(f"[NODE {node_id}] Stats: queue={queue_size}, "
                                      f"received={received}, latest_index={latest_idx}")
                            thread_report_time = current_time

                    except Timeout:
                        # Just continue on timeout - this is normal
                        continue
                    except Exception as e:
                        logger.error(f"[NODE {node_id}] Error pulling data: {e}")
                        continue

        except Exception as e:
            logger.error(f"[NODE {node_id}] Fatal error in puller thread: {e}")

        logger.info(f"[NODE {node_id}] Puller thread exiting")

    @staticmethod
    def _parse_data(data, rank):
        """Parse data received from the push socket"""
        # Extract metadata
        newline_index = data.find(b'\n')
        if newline_index != -1:
            metadata_bytes = data[:newline_index]
            data = data[newline_index + 1:]

            try:
                metadata = ast.literal_eval(metadata_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"[RANK {rank}] Error parsing metadata: {e}")
                metadata = None
        else:
            metadata = None

        # Extract tensor
        try:
            buffer = io.BytesIO(data)
            tensor_np = np.load(buffer)
            tensor = torch.from_numpy(tensor_np)
            return tensor, metadata
        except Exception as e:
            logger.error(f"[RANK {rank}] Error deserializing tensor: {e}")
            return None, None

    def __iter__(self):
        """Return an iterator over the streaming data"""
        # Get worker info for this worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        logger.info(f"[RANK {self.rank}] Worker {worker_id}/{num_workers} starting iteration")

        # Queue for this rank/worker
        node_queue = StreamingDataset._node_queues[self.node_key]
        node_stats = StreamingDataset._node_stats[self.node_key]
        stop_flag = StreamingDataset._node_stop_flags[self.node_key]

        # Iterator function
        def data_iterator():
            while not stop_flag.is_set():
                try:
                    # Try to get an item from the queue
                    try:
                        # Block with timeout
                        tensor, metadata = node_queue.get(block=True, timeout=0.1)
                    except queue.Empty:
                        # If queue is empty, check how long we've been waiting
                        if not hasattr(data_iterator, 'wait_start_time'):
                            data_iterator.wait_start_time = time.time()

                        wait_time = time.time() - data_iterator.wait_start_time

                        # If we've waited too long, raise an exception
                        if wait_time > self.config.max_wait_time:
                            logger.error(f"[RANK {self.rank}] No data received for {wait_time:.1f} seconds. Data source may be inactive.")
                            raise RuntimeError(f"Data source appears to be inactive after {wait_time:.1f} seconds")

                        # Otherwise just continue waiting
                        if wait_time > 5 and int(wait_time) % 5 == 0:  # Log every 5 seconds
                            logger.warning(f"[RANK {self.rank}] Waiting for data: {wait_time:.1f}s elapsed")

                        continue

                    # Reset wait timer since we got data
                    if hasattr(data_iterator, 'wait_start_time'):
                        delattr(data_iterator, 'wait_start_time')

                    # Convert dtype if specified
                    if self.dtype is not None:
                        tensor = tensor.to(self.dtype)

                    # Apply transformations if specified
                    if self.transforms is not None:
                        for transform in self.transforms:
                            tensor = transform(tensor)

                    # Update stats
                    with node_stats["total_yielded"].get_lock():
                        node_stats["total_yielded"].value += 1

                    # Update the highest index seen by this rank
                    if metadata and 'index' in metadata:
                        self.highest_index = max(self.highest_index, metadata['index'])

                    yield tensor

                except KeyboardInterrupt:
                    logger.info(f"[RANK {self.rank}] Worker {worker_id} received KeyboardInterrupt")
                    break
                except Exception as e:
                    logger.error(f"[RANK {self.rank}] Worker {worker_id} error: {e}")
                    # Don't continue on exceptions other than queue timeouts
                    raise

        return data_iterator()

    def get_checkpoint_info(self):
        """
        Return checkpoint info that can be saved with the model checkpoint.
        This provides hints to the data pusher about where to resume.
        """
        node_stats = StreamingDataset._node_stats[self.node_key]

        return {
            'rank': self.rank,
            'node_id': self.node_id,
            'node_address': self.node_address,  # Include address in checkpoint info
            'highest_index': self.highest_index,
            'total_samples_received': node_stats["total_received"].value,
            'total_samples_processed': node_stats["total_yielded"].value,
        }

    def close(self):
        """Clean up resources"""
        if not hasattr(self, 'node_key') or self.node_key not in StreamingDataset._node_ref_counts:
            return

        node_stats = StreamingDataset._node_stats[self.node_key]

        logger.info(f"[RANK {self.rank}] Closing streaming dataset. "
                   f"Node received {node_stats['total_received'].value} tensors, "
                   f"yielded {node_stats['total_yielded'].value} tensors, "
                   f"highest index {self.highest_index}")

        # Decrement reference counter
        with StreamingDataset._node_locks[self.node_key]:
            StreamingDataset._node_ref_counts[self.node_key] -= 1

            # If this is the last instance on this node, clean up node resources
            if StreamingDataset._node_ref_counts[self.node_key] <= 0:
                logger.info(f"[RANK {self.rank}] Last instance on node {self.node_id}, cleaning up resources")

                # Stop the puller thread
                if self.node_key in StreamingDataset._node_stop_flags:
                    StreamingDataset._node_stop_flags[self.node_key].set()

                if self.node_key in StreamingDataset._node_pullers and StreamingDataset._node_pullers[self.node_key].is_alive():
                    StreamingDataset._node_pullers[self.node_key].join(timeout=2.0)

                # Clean up resources
                if self.node_key in StreamingDataset._node_queues:
                    # Clear the queue
                    try:
                        while not StreamingDataset._node_queues[self.node_key].empty():
                            StreamingDataset._node_queues[self.node_key].get_nowait()
                    except:
                        pass

                # Remove references
                StreamingDataset._node_ref_counts.pop(self.node_key, None)
                StreamingDataset._node_pullers.pop(self.node_key, None)
                StreamingDataset._node_stop_flags.pop(self.node_key, None)
                StreamingDataset._node_queues.pop(self.node_key, None)
                StreamingDataset._node_stats.pop(self.node_key, None)

        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)
