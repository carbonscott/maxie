#!/usr/bin/env python
# -*- coding: utf-8 -*-

from maxie.datasets.psana_utils import PsanaImg
from maxie.utils.data import split_list_into_chunk

import multiprocessing
import os

import yaml
import tqdm

import hydra
from omegaconf import DictConfig

from joblib import Parallel, delayed

import logging
logging.basicConfig(level=logging.DEBUG)

def init_psana_check(exp, run, access_mode, detector_name):
    try:
        psana_img = PsanaImg(exp, run, access_mode, detector_name)
    except Exception as e:
        print(f"Failed to initialize PsanaImg: {e}!!!")

def process_batch(exp, run, access_mode, detector_name, events):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    valid_events = [event for event in tqdm.tqdm(events) if psana_img.get(event, None, 'raw') is not None]
    return valid_events

def get_psana_events(exp, run, access_mode, detector_name, num_cpus = 2, max_num_events = None, filename_postfix = None):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    num_events = len(psana_img)
    if max_num_events is not None:
        num_events = max(min(max_num_events, num_events), 0)

    batch_events = split_list_into_chunk(range(num_events), num_cpus)

    results = Parallel(n_jobs=num_cpus)(delayed(process_batch)(exp, run, access_mode, detector_name, batch) for batch in batch_events)
    valid_events = [event for batch in results for event in batch]

    output = {
        "exp"          : exp,
        "run"          : run,
        "detector_name": detector_name,
        "events"       : valid_events,
        "num_events"   : num_events,
    }

    dir_output = "outputs"
    basename_output = f"{exp}_r{run:04d}"
    if filename_postfix is not None: basename_output += filename_postfix
    file_output = f"{basename_output}.yaml"
    path_output = os.path.join(dir_output, file_output)
    os.makedirs(dir_output, exist_ok=True)

    yaml_data = yaml.dump(output)
    with open(path_output, 'w') as file:
        file.write(yaml_data)

def run_psana(exp, run, access_mode, detector_name, num_cpus = 2, max_num_events = None, filename_postfix = None):
    try:
        get_psana_events(exp, run, access_mode, detector_name, num_cpus = num_cpus, max_num_events = max_num_events, filename_postfix = filename_postfix)
    except Exception as e:
        print(f"Caught an exception: {e}!!!")

@hydra.main(config_path="hydra_config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    exp            = cfg.exp
    run            = cfg.run
    access_mode    = cfg.access_mode
    detector_name  = cfg.detector_name
    num_cpus       = cfg.num_cpus
    max_num_events = cfg.max_num_events
    postfix        = cfg.postfix

    p = multiprocessing.Process(target=init_psana_check, args=(exp, run, access_mode, detector_name))
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"Process terminated with exit code {p.exitcode}!!!")
    else:
        print(f"Psana is lauchable...")
        run_psana(exp, run, access_mode, detector_name, num_cpus, max_num_events, postfix)

if __name__ == "__main__":
    main()
