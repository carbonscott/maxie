#!/usr/bin/env python
# -*- coding: utf-8 -*-

from maxie.datasets.psana_utils import PsanaImg
from maxie.datasets.utils import split_list_into_chunk

import multiprocessing
import os
import numpy as np
import csv

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
    if data is not None:
        return np.mean(data), np.std(data)
    else:
        return np.nan, np.nan

def get_sample_stats(exp, run, access_mode, detector_name):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    num_events = len(psana_img)
    if num_events <= 10:
        return
    proposed_sample = np.random.choice(range(num_events), int(0.1*num_events), replace=False)
    print(f"Proposed sample: {proposed_sample}")
    sampled_events, means, stds = [], [], []
    for event_num in proposed_sample:
        event_mean, event_std = get_stats(psana_img, event_num)
        print(event_mean, event_std)
        means.append(event_mean)
        stds.append(event_std)
        if event_mean:
            sampled_events.append(event_num)

    output = {
        'exp' : exp,
        'run' : run,
        "detector_name": detector_name,
        "mean" : np.nanmean(means),
        "std" : np.nanstd(stds)
    }

    fields = list(output.keys())

    with open("outputs/summary_stats.csv", "a") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writerow(output)

def run_psana(exp, run, access_mode, detector_name):
    try:
        get_sample_stats(exp, run, access_mode, detector_name)
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
        run_psana(exp, run, access_mode, detector_name)

if __name__ == "__main__":
    main()
