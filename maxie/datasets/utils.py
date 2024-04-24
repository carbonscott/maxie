import numpy as np
import json

def apply_mask(data, mask, mask_value = np.nan):
    """
    Return masked data.

    Args:
        data: numpy.ndarray with the shape of (B, H, W).·
              - B: batch of images.
              - H: height of an image.
              - W: width of an image.

        mask: numpy.ndarray with the shape of (B, H, W).·

    Returns:
        data_masked: numpy.ndarray.
    """
    # Mask unwanted pixels with np.nan...
    data_masked = np.where(mask, data, mask_value)

    return data_masked




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


def load_dataset_in_json(path_json):
    """
    Sample JSON:
    [
        {
            "exp": "mfxp1002121",
            "run": 7,
            "detector_name": "Rayonix",
            "events": [500, 501],
            "num_events" : 4234
        },
        {
            "exp": "xpptut15",
            "run": 630,
            "detector_name": "jungfrau1M",
            "events": null,
            "num_events" : 4234
        },
    ]
    """
    PSANA_ACCESS_MODE = 'idx'
    with open(path_json, 'r') as file:
        entry_list = json.load(file)
        for entry in entry_list:
            exp           = entry['exp'          ]
            run           = entry['run'          ]
            detector_name = entry['detector_name']
            events        = entry['events'       ]
            num_events    = entry['num_events'   ]
            if events is None:
                events = range(num_events)
            for event in events:
                yield (exp, run, PSANA_ACCESS_MODE, detector_name, event)
