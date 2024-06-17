#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import time
import yaml
import msgpack
import argparse

from maxie.datasets.psana_dataset import PsanaDataset
from maxie.datasets.psana_utils   import PsanaImg
from maxie.utils.data             import split_list_into_chunk

# Set up MPI
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_data_tag     = 11
mpi_request_tag  = 12
mpi_terminal_tag = 13

# Define a workflow to test data loading performance...
class BenchmarkDataloader:
    def __init__(self, config, device, events):
        self.config = config
        self.device = device
        self.events = events

        self.dataloader = None

    def _initialize_dataloader(self):
        config = self.config
        device = self.device
        events = self.events

        # Psana specific...
        exp           = config['exp'          ]
        run           = config['run'          ]
        img_load_mode = config['img_load_mode']
        access_mode   = config['access_mode'  ]
        detector_name = config['detector_name']
        photon_energy = config['photon_energy']
        encoder_value = config['encoder_value']

        # Data loader specific...
        dataloader_batch_size  = config["dataloader_batch_size"]
        dataloader_num_workers = config["dataloader_num_workers"]

        # Establish dataloader...
        psana_dataset = PsanaDataset(exp, run, access_mode, detector_name, img_load_mode, events)
        dataloader = torch.utils.data.DataLoader( psana_dataset,
                                                  shuffle     = False,
                                                  pin_memory  = True,
                                                  batch_size  = dataloader_batch_size,
                                                  num_workers = dataloader_num_workers, )

        self.dataloader = dataloader

    def measure(self):
        if self.dataloader is None:
            self._initialize_dataloader()

        dataloader_iter = iter(self.dataloader)
        batch_idx       = 0
        t_s = time.monotonic()
        while True:
            try:
                batch_data, batch_metadata = next(dataloader_iter)
                batch_idx += 1
            except StopIteration:
                break
        t_e = time.monotonic()
        loading_time_in_sec = (t_e - t_s)

        return loading_time_in_sec, len(self.events)

# Let the main rank to distribute jobs and metadata...
if mpi_rank == 0:
    # [[[ ARG PARSE ]]]
    parser = argparse.ArgumentParser(description='Process a yaml file.')
    parser.add_argument('yaml', help='The input yaml file.')
    args = parser.parse_args()

    # [[[ Configure ]]]
    fl_yaml = args.yaml
    basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

    # Load the YAML file
    with open(fl_yaml, 'r') as fh:
        config = yaml.safe_load(fh)

    # ___/ USER INPUT \___
    # Psana...
    exp                  = config['exp'          ]
    run                  = config['run'          ]
    img_load_mode        = config['img_load_mode']
    access_mode          = config['access_mode'  ]
    detector_name        = config['detector_name']
    photon_energy        = config['photon_energy']
    encoder_value        = config['encoder_value']

    # Data range...
    event_min            = config['event_min']
    event_max            = config['event_max']

    # Divide events...
    num_event_groups = config["num_event_groups"]

    # Output...
    dir_results    = config["dir_results"]

    # Properties derived from user inputs...
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Derive total number of events...
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    if event_min is None: event_min = 0
    if event_max is None: event_max = len(psana_img.timestamps)

    # Split all events into event groups...
    # All MPI workers cooperate within a single event group and process all
    # groups sequentially.
    events = range(event_min, event_max)
    event_groups = split_list_into_chunk(events, max_num_chunk = num_event_groups)

    print(f"Total events of this run: {len(events)}")

    # Work through each group...
    for group_idx, event_group in enumerate(event_groups):
        # Split events into subgroups...
        event_subgroups = split_list_into_chunk(event_group, max_num_chunk = mpi_size)

        # Leave aside the chunk that will be processed by the main worker...
        num_event_subgroup_processed = 1

        # Distribute events on demand...
        while num_event_subgroup_processed < len(event_subgroups):
            # If a worker is ready???
            worker_rank = mpi_comm.recv(source=MPI.ANY_SOURCE, tag=mpi_request_tag)

            # Serialize data and then send them...
            events = list(event_subgroups[worker_rank])
            data_to_transfer = {
                "config"    : config,
                "device"    : device,
                "events"    : events,
                "group_idx" : group_idx,
            }
            serialized_data_to_transfer = msgpack.packb(data_to_transfer)
            mpi_comm.send(serialized_data_to_transfer, dest = worker_rank, tag = mpi_data_tag)

            num_event_subgroup_processed += 1

        # Process data...
        events = list(event_subgroups[mpi_rank])
        print(f"Rank {mpi_rank}: Event group {group_idx:02d}, Total events {len(events)}, First sample event {events[0]:06d}, Begins processing...")
        benchmark_dataloader = BenchmarkDataloader(config, device, events)
        loading_time, num_events = benchmark_dataloader.measure()
        print(f"Rank {mpi_rank}: Event group {group_idx:02d}, Total events {len(events)}, Total time {loading_time:.2f} s, Average time {loading_time/num_events * 1e3:.2f} ms/event...")

    # Done and send terminal signal (None) to all workers
    for i in range(1, mpi_size):
        serialized_data_to_transfer = msgpack.packb(None)
        mpi_comm.send(serialized_data_to_transfer, dest=i, tag=mpi_terminal_tag)
else:
    while True:
        # Request works from the main worker...
        mpi_comm.send(mpi_rank, dest=0, tag=mpi_request_tag)

        # Get data distributed from the main and de-serialize them...
        serialized_data_recv = mpi_comm.recv(source = 0, tag = MPI.ANY_TAG)
        data_recv = msgpack.unpackb(serialized_data_recv)

        # Break the look when encountering a terminal signal???
        if data_recv is None:
            break

        # Unpack data
        config    = data_recv["config"]
        device    = data_recv["device"]
        events    = data_recv["events"]
        group_idx = data_recv["group_idx"]

        # Process data...
        print(f"Rank {mpi_rank}: Event group {group_idx:02d}, Total events {len(events)}, First sample event {events[0]:06d}, Begins processing...")
        benchmark_dataloader = BenchmarkDataloader(config, device, events)
        loading_time, num_events = benchmark_dataloader.measure()
        print(f"Rank {mpi_rank}: Event group {group_idx:02d}, Total events {len(events)}, Total time {loading_time:.2f} s, Average time {loading_time/num_events * 1e3:.2f} ms/event...")
