import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from milp import milp
from metrics import calc_prec_recall
from plots import plot_prec_recall
from parse import *


class Evaluator():

    def __init__(self, distances, predictions, objectives) -> None:
        self.distances = distances
        self.predictions = predictions
        self.objectives = objectives

    def update(self, method, features, mu, bag_to_index):        
        mu_normalized = mu / norm(mu)
        distances = (1 - features @ mu_normalized).flatten()

        predictions, objective = predict(bag_to_index, distances, features, mu)
        
        self.distances[method].extend(distances)
        self.predictions[method].extend(predictions)
        self.objectives[method].append(objective)


def filter_by_counts(subj_bag_indices, min_counts=1, max_counts=np.inf):

    filtered = []
    for bag_indices in subj_bag_indices.values():
        
        for indices in bag_indices.values():

            indices_len = len(indices)

            if min_counts <= indices_len <= max_counts:
                filtered.extend(list(indices))

    return filtered

def predict(bag_to_index, dist, X, mu):
    y_pred = np.zeros(len(X))

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

    indices = filter_by_counts(subj_bag_indices, 2, 10)

    metadata = metadata[indices]
    features = features[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    labels = metadata[:, 7].astype(int)

    dists = {}
    preds = {}
    objs = {}

    methods = ["method_*", "method_1", "method_2", "method_3", "method_4"]
    for m in methods:
        dists[m] = []
        preds[m] = []
        objs[m] = []

    y_true = []
    # sorted_subjects = []

    # subjects_counter = 0

    evaluator = Evaluator(dists, preds, objs)

    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        
        m = len(idx)
        if m >= 15:
            continue

        counter += 1
        print(counter)

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = labels[idx]
        
        y_true.extend(labels_subset)

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        img_per_subj = len(unique_paths)

        # sorted_subjects.append(np.arange(subjects_counter, subjects_counter + m))
        # subjects_counter += m

        # Baseline *: Mean

        mu = np.mean(features_subset[labels_subset.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_*', features_subset, mu, bag_to_index)
        
        # Baseline 1: Median
        
        mu = np.median(features_subset, axis=0, keepdims=True).T
        mu_normalized = mu / norm(mu)
        cos_dist = (1 - features_subset @ mu_normalized).flatten()
        y_pred, objective = predict(bag_to_index, cos_dist, features_subset, mu)
        evaluator.update('method_1', features_subset, mu, bag_to_index)
        y_pred_method_1 = y_pred

        # Baseline 2: Two Pass Median

        mu = np.median(features_subset[y_pred_method_1.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_2', features_subset, mu, bag_to_index)

        # Baseline 3: Average of Medians

        mu = np.mean(features_subset[y_pred_method_1.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_3', features_subset, mu, bag_to_index)

        # Baseline 4: MILP

        # '''
        K = list(index_to_bag.values())
        mu = milp(features_subset, K, False)
        mu = np.atleast_2d(mu).T
        evaluator.update('method_4', features_subset, mu, bag_to_index)
        # '''

    # generate plots

    for m in methods:
        label = m.replace('_', ' ')

        if len(preds[m]) == 0:
            continue

        recall, prec = calc_prec_recall(y_true, preds[m], dists[m])
        plt.plot(recall, prec, label=label)

        print(f'{label}: {np.mean(objs[m]):.6f}')

    plot_prec_recall()
