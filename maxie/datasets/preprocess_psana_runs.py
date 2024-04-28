#!/usr/bin/env python
# -*- coding: utf-8 -*-

from maxie.datasets.psana_utils import PsanaImg
from maxie.datasets.utils import split_list_into_chunk

import multiprocessing
import os

import yaml
import tqdm

import hydra
from omegaconf import DictConfig

import ray

def init_psana_check(exp, run, access_mode, detector_name):
    try:
        psana_img = PsanaImg(exp, run, access_mode, detector_name)
    except Exception as e:
        print(f"Failed to initialize PsanaImg: {e}!!!")

@ray.remote
def process_batch(exp, run, access_mode, detector_name, events):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    valid_events = [event for event in tqdm.tqdm(events) if psana_img.get(event, None, 'calib') is not None]
    return valid_events

def get_psana_events(exp, run, access_mode, detector_name, num_cpus = 2):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    num_events = len(psana_img)

    batch_events = split_list_into_chunk(list(range(num_events)), num_cpus)

    ray.init(num_cpus = num_cpus)

    batch_futures = [process_batch.remote(exp, run, access_mode, detector_name, batch) for batch in batch_events]
    results = ray.get(batch_futures)
    valid_events = [event for batch in results for event in batch]

    ## true_events = [ event for event in tqdm.tqdm(range(num_events)) if psana_img.get(event, None, 'calib') is not None ]

    output = {
        'exp' : exp,
        'run' : run,
        "detector_name": detector_name,
        "events": valid_events,
        "num_events" : num_events
    }

    dir_output  = 'outputs'
    file_output = f'{exp}_r{run:04d}.yaml'
    path_output = os.path.join(dir_output, file_output)
    os.makedirs(dir_output, exist_ok=True)

    yaml_data = yaml.dump(output)
    with open(path_output, 'w') as file:
        file.write(yaml_data)

    ray.shutdown()

def run_psana(exp, run, access_mode, detector_name, num_cpus = 2):
    try:
        get_psana_events(exp, run, access_mode, detector_name, num_cpus = num_cpus)
    except Exception as e:
        print(f"Caught an exception: {e}!!!")

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    exp           = cfg.exp
    run           = cfg.run
    access_mode   = cfg.access_mode
    detector_name = cfg.detector_name
    num_cpus      = cfg.num_cpus

    p = multiprocessing.Process(target=init_psana_check, args=(exp, run, access_mode, detector_name))
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"Process terminated with exit code {p.exitcode}!!!")
    else:
        print(f"Psana is lauchable...")
        run_psana(exp, run, access_mode, detector_name, num_cpus)

if __name__ == "__main__":
    main()
