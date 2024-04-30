#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import json
import tqdm
import argparse
import random

from joblib import Parallel, delayed
from maxie.utils.data import split_list_into_chunk, split_dataset

parser = argparse.ArgumentParser(
    description="Generate a JSON dataset file."
)
parser.add_argument('--yaml', type = str, help='Path to the YAML dataset file')
parser.add_argument('--num_cpus', type = int, help='Number of cpus used in processing these files.')
parser.add_argument('--dir_output', type = str, help='Direcotry to save the output json files.')
parser.add_argument('--train_frac', type = float, default = 0.8, help='Direcotry to save the output json files. (Default: 0.8)')
parser.add_argument('--seed',  type = lambda s: int(s) if s.isdigit() else None, default = None, help='Seed for shuffling the output.  (Default: None; no shuffle)')
args = parser.parse_args()

def safe_load_yaml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def process_batch(batch_file, batch_idx):
    return [ safe_load_yaml(file) for file in tqdm.tqdm(batch_file, desc=f"Processing batch [{batch_idx}]") ]

path_yaml  = args.yaml
num_cpus   = args.num_cpus
dir_output = args.dir_output
train_frac = args.train_frac
seed       = args.seed

if seed is not None: random.seed(seed)

# Read the dataset.yaml
yaml_files = safe_load_yaml(path_yaml)
if seed is not None:
    print(f"Dataset is shuffled with the seed {seed}.")
    random.shuffle(yaml_files)

batches = split_list_into_chunk(yaml_files, args.num_cpus)
results = Parallel(n_jobs=num_cpus)(delayed(process_batch)(batch, batch_idx) for batch_idx, batch in enumerate(batches))
dataset = [event for batch in results for event in batch]

dataset_train, dataset_eval = split_dataset(dataset, train_frac)

os.makedirs(dir_output, exist_ok=True)
file_yaml        = os.path.basename(path_yaml)
dataset_basename = file_yaml[:file_yaml.rfind('.yaml')]
path_json_train  = os.path.join(dir_output, f"{dataset_basename}.train.json")
path_json_eval   = os.path.join(dir_output, f"{dataset_basename}.eval.json")

with open(path_json_train, 'w') as file:
    json.dump(dataset_train, file)
with open(path_json_eval, 'w') as file:
    json.dump(dataset_eval, file)
