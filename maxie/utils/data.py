#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import more_itertools

from math import ceil

import logging

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


def split_list_into_chunk(input_list, max_num_chunk = 2):
    '''
    [1, 2, 3, 4, 5, 6], 2 -> [iter([1, 2, 3]), iter(4, 5, 6)]

    For splitting a dictionary, users can turn the dictionary into a list of
    tuples using input_dict.items().
    '''
    return more_itertools.divide(max_num_chunk, input_list)
