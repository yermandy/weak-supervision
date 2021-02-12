import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from milp import milp
from metrics import calc_prec_recall
from plots import plot_prec_recall
from parse import *


def filter_by_counts(subj_bag_indices, min_counts=1, max_counts=np.inf):

    filtered = []
    for bag_indices in subj_bag_indices.values():
        
        for indices in bag_indices.values():

            indices_len = len(indices)

            if min_counts <= indices_len <= max_counts:
                filtered.extend(list(indices))

    return filtered

def predict(bag_to_index, m, dist, X, mu):
    y_pred = np.zeros(m)

    objective = []

    for indices in bag_to_index.values():
        argmin = dist[indices].argmin()
        pred_id = indices[argmin]
        y_pred[pred_id] = 1
        
        d = mu - X[indices].T
        objective.append(np.abs(d).sum(axis=0).min())
    
    objective = np.mean(objective)

    return y_pred, objective


if __name__ == "__main__":
    features = np.load("resources/features/ijbb_features.npy")

    metadata_file = "resources/ijbb_metadata.csv"
    metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",")

    subj_bag_indices = parse_metadata(metadata)

    indices = filter_by_counts(subj_bag_indices, 2)

    metadata = metadata[indices]
    features = features[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    ground_true = metadata[:, 7].astype(int)

    dists = {}
    preds = {}
    objs = {}

    baselines = ["baseline_*", "baseline_1", "baseline_2", "baseline_3", "baseline_4"]
    for b in baselines:
        dists[b] = []
        preds[b] = []
        objs[b] = []

    y_true = []
    # sorted_subjects = []

    # subjects_counter = 0
    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        
        m = len(idx)
        if m >= 7:
            continue

        counter += 1
        print(counter)

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = ground_true[idx]
        
        y_true.extend(labels_subset)

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        img_per_subj = len(unique_paths)

        # sorted_subjects.append(np.arange(subjects_counter, subjects_counter + m))
        # subjects_counter += m

        # Baseline *: Mean

        mu = np.mean(features_subset[labels_subset.astype(bool)], axis=0, keepdims=True).T
        mu_normalized = mu / norm(mu)
        cos_dist = (1 - features_subset @ mu_normalized).flatten()

        y_pred, objective = predict(bag_to_index, m, cos_dist, features_subset, mu)

        b = 'baseline_*'
        dists[b].extend(cos_dist)
        preds[b].extend(y_pred)
        objs[b].append(objective)
        
        # Baseline 1: Median
        
        mu = np.median(features_subset, axis=0, keepdims=True).T
        mu_normalized = mu / norm(mu)
        cos_dist = (1 - features_subset @ mu_normalized).flatten()

        y_pred, objective = predict(bag_to_index, m, cos_dist, features_subset, mu)

        b = 'baseline_1'
        dists[b].extend(cos_dist)
        preds[b].extend(y_pred)
        objs[b].append(objective)

        y_pred_baseline_1 = y_pred

        # Baseline 2: Two Pass Median

        mu = np.median(features_subset[y_pred_baseline_1.astype(bool)], axis=0, keepdims=True).T
        mu_normalized = mu / norm(mu)
        cos_dist = (1 - features_subset @ mu_normalized).flatten()

        y_pred, objective = predict(bag_to_index, m, cos_dist, features_subset, mu)

        b = 'baseline_2'
        dists[b].extend(cos_dist)
        preds[b].extend(y_pred)
        objs[b].append(objective)

        # Baseline 3: Average of Medians

        mu = np.mean(features_subset[y_pred_baseline_1.astype(bool)], axis=0, keepdims=True).T
        mu_normalized = mu / norm(mu)
        cos_dist = (1 - features_subset @ mu_normalized).flatten()

        y_pred, objective = predict(bag_to_index, m, cos_dist, features_subset, mu)

        b = 'baseline_3'
        dists[b].extend(cos_dist)
        preds[b].extend(y_pred)
        objs[b].append(objective)

        # Baseline 4: MILP

        '''
        K = list(index_to_bag.values())
        mu, milp_obj_val = milp(features_subset, K, False, True)

        mu = np.atleast_2d(mu).T
        mu_normalized = mu / norm(mu)

        cos_dist = (1 - features_subset @ mu_normalized).flatten()

        y_pred, objective = predict(bag_to_index, m, cos_dist, features_subset, mu)

        b = 'baseline_4'
        dists[b].extend(cos_dist)
        preds[b].extend(y_pred)
        objs[b].append(milp_obj_val / img_per_subj)
        # '''

    # generate plots

    for b in baselines:
        label = b.replace('_', ' ')

        if len(preds[b]) == 0:
            continue

        recall, prec = calc_prec_recall(y_true, preds[b], dists[b])
        plt.plot(recall, prec, label=label)

        print(f'{label}: {np.mean(objs[b]):.6f}')

    plot_prec_recall()
