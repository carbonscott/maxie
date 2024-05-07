#!/usr/bin/env python
# -*- coding: utf-8 -*-

from maxie.datasets.psana_utils import PsanaImg
from maxie.datasets.utils import split_list_into_chunk

import multiprocessing
import os
import numpy as np

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

def get_stats(img, event_num):
    data = img.get(event_num, mode = "calib")
    return np.mean(data), np.std(data)

def get_sample_stats(exp, run, detector_name):
    psana_img = PsanaImg(exp, run, "calib", detector_name)
    num_events = len(psana_img)
    valid_events = [event for event in range(num_events) if event is not None]
    sampled_events = np.random.choice(valid_events, len(valid_events)//10, replace=False)
    means, stds = np.zeros(len(sampled_events)), np.zeros(len(sampled_events))
    for i, event_num in enumerate(sampled_events):
        means[i] = get_stats(psana_img, event_num)
        stds[i] = get_stats(psana_img, event_num)

    output = {
        'exp' : exp,
        'run' : run,
        "detector_name": detector_name,
        "events": valid_events,
        "num_events" : num_events,
        "mean" : np.mean(means),
        "std" : np.std(stds)
    }

    dir_output  = 'outputs'
    file_output = f'{exp}_r{run:04d}_stats.yaml'
    path_output = os.path.join(dir_output, file_output)
    os.makedirs(dir_output, exist_ok=True)

    yaml_data = yaml.dump(output)
    with open(path_output, 'w') as file:
        file.write(yaml_data)

def run_psana(exp, run, detector_name):
    try:
        get_sample_stats(exp, run, detector_name)
    except Exception as e:
        print(f"Caught an exception: {e}!!!")

@hydra.main(config_path="hydra_config", config_name="base", version_base=None)
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
        run_psana(exp, run, detector_name)

if __name__ == "__main__":
    main()
