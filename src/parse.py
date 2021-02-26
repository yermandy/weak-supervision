from typing import Dict
import numpy as np


def open_metadata(metadata_file: str):
    return np.genfromtxt(metadata_file, dtype=str, delimiter=",", skip_header=1)


def parse_metadata(metadata) -> Dict[int, Dict[str, int]]:
    """ {subject: {bag: [indices]}} """

    bags = metadata[:, 0]
    subj = metadata[:, 6].astype(int)

    subj_bag_indices = {}

    for i, (b, s) in enumerate(zip(bags, subj)):

        if s not in subj_bag_indices:
            subj_bag_indices[s] = {}

        if b not in subj_bag_indices[s]:
            subj_bag_indices[s][b] = []

        subj_bag_indices[s][b].append(i)

    return subj_bag_indices


def filter_by_counts(subj_bag_indices, min_counts=1, max_counts=np.inf):

    filtered = []
    for bag_indices in subj_bag_indices.values():
        
        for indices in bag_indices.values():

            indices_len = len(indices)

            if min_counts <= indices_len <= max_counts:
                filtered.extend(list(indices))

    return filtered
    

def parse_paths(paths):
    path_to_bag = {}
    # index (int) -> bag (int)
    index_to_bag = {}
    # bag (int) -> indices (int array)
    bag_to_index = {}
    
    # Encode unique paths (str -> int)
    bag_id = 0
    for path in paths:
        if path not in path_to_bag:
            path_to_bag[path] = bag_id
            bag_id += 1

    for index, path in enumerate(paths):
        bag = path_to_bag[path]
        index_to_bag[index] = bag
        
        if bag not in bag_to_index:
            bag_to_index[bag] = [index]
        else:
            bag_to_index[bag].append(index)

    unique_paths = list(path_to_bag.keys())
    return unique_paths, bag_to_index, index_to_bag
