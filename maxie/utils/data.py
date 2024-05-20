#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
from sklearn.model_selection import train_test_split
import re

logger = logging.getLogger(__name__)

def split_dataset(dataset_list, fracA, seed = None):
    ''' Split a dataset into two subsets A and B by user-specified fraction.
    '''
    # Set seed for data spliting...
    if seed is not None:
        random.seed(seed)

    # Indexing elements in the dataset...
    size_dataset = len(dataset_list)
    idx_dataset = range(size_dataset)

    # Get the size of the dataset and the subset A...
    size_fracA   = int(fracA * size_dataset)

    # Randomly choosing examples for constructing subset A...
    idx_fracA_list = random.sample(idx_dataset, size_fracA)

    # Obtain the subset B...
    idx_fracB_list = set(idx_dataset) - set(idx_fracA_list)
    idx_fracB_list = sorted(list(idx_fracB_list))

    fracA_list = [ dataset_list[idx] for idx in idx_fracA_list ]
    fracB_list = [ dataset_list[idx] for idx in idx_fracB_list ]

    return fracA_list, fracB_list

def split_dataset_stratified(dataset_list, fracA, detectors_by_exp, yaml_regex=r"/(\w+)_r\d+\.yaml"):
    """Split a dataset into two subsets A and B by user-specified fraction, maintaining a constant proportion of 
    samples from each detector type in each sample. 
    """
    # For each file in the dataset, determine the detector used to generate data
    detector_list = [run["detector_name"] for run in dataset_list]
    # Then, perform stratified split
    fracA_list, fracB_list = train_test_split(dataset_list, train_size=fracA, stratify=detector_list)
    return fracA_list, fracB_list

def split_list_into_chunk(input_list, max_num_chunk = 2):

    chunk_size = len(input_list) // max_num_chunk + 1

    size_list = len(input_list)

    chunked_list = []
    for idx_chunk in range(max_num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        ## if idx_chunk == max_num_chunk - 1: idx_e = len(input_list)
        if idx_e >= size_list: idx_e = size_list

        seg = input_list[idx_b : idx_e]
        chunked_list.append(seg)

        if idx_e == size_list: break

    return chunked_list


def split_dict_into_chunk(input_dict, max_num_chunk = 2):

    chunk_size = len(input_dict) // max_num_chunk + 1

    size_dict = len(input_dict)
    kv_iter   = iter(input_dict.items())

    chunked_dict_in_list = []
    for idx_chunk in range(max_num_chunk):
        chunked_dict = {}
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_e >= size_dict: idx_e = size_dict

        for _ in range(idx_e - idx_b):
            k, v = next(kv_iter)
            chunked_dict[k] = v
        chunked_dict_in_list.append(chunked_dict)

        if idx_e == size_dict: break

    return chunked_dict_in_list
