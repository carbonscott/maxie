import torch
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict
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
import json
import yaml
import math
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class StreamingDataConfig:
    C: int                                      # Number of channels
    H: int                                      # Height of tensor
    W: int                                      # Width of tensor
    address: Optional[str] = None               # Single address (legacy support)
    addresses: List[str] = None                 # List of addresses to pull data from
    address_assignment: str = "round-robin"     # How to assign addresses to nodes
    node_address_map: Dict[int, List[str]] = None  # Explicit node->address mapping
    sockets_per_node: int = 1                   # Number of sockets per node (for multi-socket approaches)
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

        # Validate the number of sockets per node
        if self.sockets_per_node <= 0:
            raise ValueError("sockets_per_node must be greater than 0")

        # Validate the address_assignment strategy
        valid_strategies = ["round-robin", "random", "explicit", "locality-aware"]
        if self.address_assignment not in valid_strategies:
            raise ValueError(f"address_assignment must be one of {valid_strategies}")

    def get_addresses_for_node(self, node_id):
        """Determine which addresses this node should connect to.

        Returns:
            List[str]: A list of addresses this node should connect to.
        """
        if not self.addresses:
            raise ValueError("No addresses configured")

        # If explicit mapping is provided and this node is in the map
        if self.address_assignment == "explicit" and node_id in self.node_address_map:
            addrs = self.node_address_map[node_id]
            if isinstance(addrs, str):
                return [addrs]  # Convert single address to list
            return addrs[:self.sockets_per_node]  # Limit to specified number of sockets

        # If only one address is available, all nodes use it
        if len(self.addresses) == 1:
            return [self.addresses[0]]

        # Number of addresses to assign (limited by available addresses)
        num_addresses = min(self.sockets_per_node, len(self.addresses))

        # Multiple addresses - use assignment strategy
        if self.address_assignment == "random":
            # Use a seeded random assignment based on node_id
            import random
            random_gen = random.Random(node_id)
            return random_gen.sample(self.addresses, num_addresses)

        elif self.address_assignment == "locality-aware":
            # This would use network topology or IP address proximity
            # For now, implement a simple version that groups addresses by subnet
            # (actual implementation would need more sophisticated network knowledge)

            # Extract host part from addresses
            hosts = []
            for addr in self.addresses:
                if addr.startswith('tcp://'):
                    host = addr.split('//')[1].split(':')[0]
                    hosts.append(host)
                else:
                    hosts.append(addr)  # Can't parse, use as is

            # Get this node's hostname
            node_hostname = socket.gethostname()

            # Try to get IP address
            try:
                node_ip = socket.gethostbyname(node_hostname)
            except:
                node_ip = "127.0.0.1"  # Default if can't resolve

            # Score addresses by "closeness" to this node
            # (This is a very simple heuristic - would need to be replaced with actual network topology)
            def score_address(addr_idx):
                host = hosts[addr_idx]
                if host == node_hostname or host == node_ip:
                    return 100  # Same host gets highest score
                if host.startswith('127.0.0.') or host == 'localhost':
                    return 90   # Local addresses get high score
                return 0        # Default score

            # Sort addresses by score (descending)
            scored_addresses = [(score_address(i), i) for i in range(len(self.addresses))]
            scored_addresses.sort(reverse=True)

            # Take top addresses up to sockets_per_node
            selected_indices = [idx for _, idx in scored_addresses[:num_addresses]]
            return [self.addresses[idx] for idx in selected_indices]

        else:  # Default to "round-robin"
            # More sophisticated round-robin that gives each node a chunk of consecutive addresses
            total_addresses = len(self.addresses)

            # If we have more addresses than nodes*sockets_per_node, distribute them evenly
            if total_addresses >= self.num_nodes * self.sockets_per_node:
                start_idx = (node_id * self.sockets_per_node) % total_addresses
                result = []
                for i in range(num_addresses):
                    idx = (start_idx + i) % total_addresses
                    result.append(self.addresses[idx])
                return result
            else:
                # Otherwise, basic round-robin starting at node's index
                start_idx = node_id % total_addresses
                result = []
                for i in range(num_addresses):
                    idx = (start_idx + i) % total_addresses
                    result.append(self.addresses[idx])
                return result


class SocketHandler:
    """Manages a connection to a data source via a socket"""

    def __init__(self, address, timeout_ms, node_id, rank, node_queue, stop_flag, node_stats):
        self.address = address
        self.timeout_ms = timeout_ms
        self.node_id = node_id
        self.rank = rank
        self.node_queue = node_queue
        self.stop_flag = stop_flag
        self.node_stats = node_stats
        self.thread = None
        self.connected = False
        self.last_report_time = time.time()

    def start(self):
        """Start the puller thread for this socket"""
        self.thread = threading.Thread(
            target=self._pull_data_thread,
            args=(
                self.address,
                self.timeout_ms,
                self.node_id,
                self.rank,
                self.node_queue,
                self.stop_flag,
                self.node_stats,
            )
        )
        self.thread.daemon = True
        self.thread.start()

    def is_alive(self):
        """Check if the thread is alive"""
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout=None):
        """Join the thread"""
        if self.thread is not None:
            self.thread.join(timeout)

    def _pull_data_thread(self, address, timeout_ms, node_id, rank, node_queue, stop_flag, node_stats):
        """Background thread to continuously pull data and fill the queue"""
        # Make sure we have our own report time for the thread
        thread_report_time = time.time()
        socket_stats = {
            'received': 0,
            'queue_full': 0,
            'timeouts': 0,
            'errors': 0
        }

        try:
            with Pull0(dial=address) as sock:
                sock.recv_timeout = timeout_ms
                self.connected = True

                logger.info(f"[NODE {node_id}] Puller initialized, dialing to {address}")

                while not stop_flag.is_set():
                    try:
                        # Pull data
                        data = sock.recv()
                        socket_stats['received'] += 1

                        # Track received data
                        with node_stats["total_received"].get_lock():
                            node_stats["total_received"].value += 1

                        # Extract tensor and metadata
                        tensor, metadata = StreamingDataset._parse_data(data, rank)

                        if tensor is not None:
                            # Put in the queue (block if full)
                            try:
                                node_queue.put((tensor, metadata, address), block=True, timeout=1.0)
                            except queue.Full:
                                socket_stats['queue_full'] += 1
                                logger.warning(f"[NODE {node_id}][SOCKET {address}] Queue full, dropping tensor")
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

                            logger.info(f"[NODE {node_id}][SOCKET {address}] Stats: queue={queue_size}, "
                                      f"socket_received={socket_stats['received']}, "
                                      f"queue_full={socket_stats['queue_full']}, "
                                      f"node_received={received}, latest_index={latest_idx}")
                            thread_report_time = current_time

                    except Timeout:
                        # Just continue on timeout - this is normal
                        socket_stats['timeouts'] += 1
                        continue
                    except Exception as e:
                        socket_stats['errors'] += 1
                        logger.error(f"[NODE {node_id}][SOCKET {address}] Error pulling data: {e}")
                        continue

        except Exception as e:
            logger.error(f"[NODE {node_id}][SOCKET {address}] Fatal error in puller thread: {e}")

        logger.info(f"[NODE {node_id}][SOCKET {address}] Puller thread exiting")
        self.connected = False


class StreamingDataset(IterableDataset):
    """
    A streaming dataset for multi-node training that:
    - Supports multiple socket connections per node
    - Has one queue per node (shared by all ranks on that node)
    - Waits instead of generating random tensors if data isn't available
    - Tracks global indices for simple checkpointing/resumption
    - Works with distributed training across multiple nodes
    """

    # Class-level variables to manage node-level connections
    _node_queues = {}          # Shared queues per node
    _node_sockets = {}         # Socket handlers per node (multiple per node possible)
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

        # Get addresses for this node
        self.node_addresses = config.get_addresses_for_node(self.node_id)

        if not self.node_addresses:
            raise ValueError(f"No addresses assigned to node {self.node_id}")

        logger.info(f"[RANK {self.rank}] Node {self.node_id} assigned addresses: {self.node_addresses}")

        # Create a unique key for this node
        self.node_key = f"node_{self.node_id}"

        # Initialize reporting variables
        self.last_report_time = time.time()
        self.report_interval = 10.0  # seconds

        # Track highest global index seen by this rank
        self.highest_index = -1

        # Track performance metrics per address
        self.address_metrics = {addr: {'count': 0, 'latency': 0.0} for addr in self.node_addresses}

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

                # Create a stop flag for the puller threads
                StreamingDataset._node_stop_flags[self.node_key] = mp.Event()

                # Create shared statistics counters
                StreamingDataset._node_stats[self.node_key] = {
                    'total_received': mp.Value('i', 0),
                    'total_yielded': mp.Value('i', 0),
                    'latest_index': mp.Value('i', -1),
                }

                # Initialize reference counter
                StreamingDataset._node_ref_counts[self.node_key] = 0

                # Initialize the socket handlers dictionary for this node
                StreamingDataset._node_sockets[self.node_key] = {}

                # Start socket handlers for this node
                self._start_node_sockets()

            # Increment the reference counter
            StreamingDataset._node_ref_counts[self.node_key] += 1

            # Log node resource status
            logger.info(f"[RANK {self.rank}] Using node-level resources for node {self.node_id}, "
                      f"ref count = {StreamingDataset._node_ref_counts[self.node_key]}, "
                      f"connected to {len(StreamingDataset._node_sockets[self.node_key])} sockets")

    def _start_node_sockets(self):
        """Start socket handlers for this node"""
        node_queue = StreamingDataset._node_queues[self.node_key]
        node_stats = StreamingDataset._node_stats[self.node_key]
        stop_flag = StreamingDataset._node_stop_flags[self.node_key]

        # Start a handler for each address
        for address in self.node_addresses:
            if address in StreamingDataset._node_sockets[self.node_key]:
                # Skip if we already have a handler for this address
                continue

            # Log that we're starting a socket handler
            logger.info(f"[RANK {self.rank}] Starting socket handler for node {self.node_id}, address {address}")

            # Create and start the socket handler
            handler = SocketHandler(
                address,
                self.config.timeout_ms,
                self.node_id,
                self.rank,
                node_queue,
                stop_flag,
                node_stats
            )

            StreamingDataset._node_sockets[self.node_key][address] = handler
            handler.start()

        # Wait for initial connections
        start_time = time.time()
        active_sockets = 0

        logger.info(f"[RANK {self.rank}] Waiting for initial connections...")

        while time.time() - start_time < self.config.connect_timeout and active_sockets == 0:
            # Count active connections
            active_sockets = sum(1 for handler in StreamingDataset._node_sockets[self.node_key].values()
                               if handler.connected)

            # If we have data or at least one connection, we're good to go
            try:
                if active_sockets > 0 or not node_queue.empty():
                    break
            except Exception as e:
                logger.warning(f"[RANK {self.rank}] Error checking queue: {e}")

            time.sleep(0.1)

        logger.info(f"[RANK {self.rank}] Connected to {active_sockets}/{len(self.node_addresses)} sockets")

        if active_sockets == 0:
            logger.warning(f"[RANK {self.rank}] No connections established after {self.config.connect_timeout}s")

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

        # Counter for address metrics
        address_counts = Counter()

        # Iterator function
        def data_iterator():
            while not stop_flag.is_set():
                try:
                    # Try to get an item from the queue
                    try:
                        # Block with timeout
                        start_time = time.time()
                        tensor, metadata, address = node_queue.get(block=True, timeout=0.1)
                        get_time = time.time() - start_time

                        # Update address metrics
                        address_counts[address] += 1

                        # Every 1000 items, log the distribution of data from different sockets
                        if sum(address_counts.values()) % 1000 == 0:
                            logger.info(f"[RANK {self.rank}] Data source distribution: {dict(address_counts)}")

                    except queue.Empty:
                        # If queue is empty, check how long we've been waiting
                        if not hasattr(data_iterator, 'wait_start_time'):
                            data_iterator.wait_start_time = time.time()

                        wait_time = time.time() - data_iterator.wait_start_time

                        # If we've waited too long, raise an exception
                        if wait_time > self.config.max_wait_time:
                            logger.error(f"[RANK {self.rank}] No data received for {wait_time:.1f} seconds. Data source may be inactive.")

                            # Check if we have any active connections
                            active_sockets = sum(1 for handler in StreamingDataset._node_sockets[self.node_key].values()
                                             if handler.is_alive() and handler.connected)

                            if active_sockets == 0:
                                raise RuntimeError(f"All data sources appear to be inactive after {wait_time:.1f} seconds")
                            else:
                                # Some connections are active but no data is coming through
                                raise RuntimeError(f"Data sources appear to be stalled after {wait_time:.1f} seconds")

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
            'node_addresses': self.node_addresses,
            'highest_index': self.highest_index,
            'total_samples_received': node_stats["total_received"].value,
            'total_samples_processed': node_stats["total_yielded"].value,
            'socket_distribution': dict(self.address_metrics),
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

                # Stop the puller threads
                if self.node_key in StreamingDataset._node_stop_flags:
                    StreamingDataset._node_stop_flags[self.node_key].set()

                # Join all socket threads
                if self.node_key in StreamingDataset._node_sockets:
                    for address, handler in StreamingDataset._node_sockets[self.node_key].items():
                        logger.info(f"[RANK {self.rank}] Joining thread for address {address}")
                        handler.join(timeout=2.0)

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
                StreamingDataset._node_sockets.pop(self.node_key, None)
                StreamingDataset._node_stop_flags.pop(self.node_key, None)
                StreamingDataset._node_queues.pop(self.node_key, None)
                StreamingDataset._node_stats.pop(self.node_key, None)

        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)
