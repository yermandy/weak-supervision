from typing import List
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from milp import milp
from metrics import calc_prec_recall
from plots import plot_prec_recall
from parse import *
from sklearn.decomposition import PCA


class Evaluator():

    def __init__(self, methods: List[str] = None) -> None:
        self.distances = {}
        self.predictions = {}
        self.objectives = {}

        if methods is not None:
            for method in methods:
                self.add_method(method)


    def get_methods(self):
        return list(self.distances.keys())


    def add_method(self, method):
        self.distances[method] = []
        self.predictions[method] = []
        self.objectives[method] = []


    def update(self, method, features, mu, bag_to_index):        
        if method not in self.distances:
            self.add_method(method)

        mu_normalized = mu / norm(mu)
        distances = (1 - features @ mu_normalized).flatten()

        predictions, objective = predict(bag_to_index, distances, features, mu)
        
        self.distances[method].extend(distances)
        self.predictions[method].extend(predictions)
        self.objectives[method].append(objective)
        
        return distances, predictions, objective


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

    indices = filter_by_counts(subj_bag_indices, 2)

    metadata = metadata[indices]
    features = features[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    labels = metadata[:, 7].astype(int)

    y_true = []
    # sorted_subjects = []

    # subjects_counter = 0

    evaluator = Evaluator()

    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        
        m = len(idx)
        if m >= 30:
            continue

        counter += 1
        print(counter)

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = labels[idx]

        y_true.extend(labels_subset)

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        bags_number = len(unique_paths)

        # sorted_subjects.append(np.arange(subjects_counter, subjects_counter + m))
        # subjects_counter += m

        # Method *: Mean

        mu = np.median(features_subset[labels_subset.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_*', features_subset, mu, bag_to_index)
        
        # Method 1: Median
        
        mu = np.median(features_subset, axis=0, keepdims=True).T
        distances, predictions_method_1, objective = evaluator.update('method_1', features_subset, mu, bag_to_index)

        # Method 2: Two-Stage Median

        mu = np.median(features_subset[predictions_method_1.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_2', features_subset, mu, bag_to_index)

        # Method 3: Average of Medians

        mu = np.mean(features_subset[predictions_method_1.astype(bool)], axis=0, keepdims=True).T
        evaluator.update('method_3', features_subset, mu, bag_to_index)

        # Method 4: MILP

        # '''
        n_components = min(len(features_subset), 2)
        # n_components = 1
        pca = PCA(n_components)
        features_reduced = pca.fit_transform(features_subset)
        K = list(index_to_bag.values())
        mu, alphas = milp(features_reduced, K, False, return_alphas=True)
        mu = np.atleast_2d(mu).T
        distances, predictions, objective = evaluator.update('method_4', features_reduced, mu, bag_to_index)
        
        mu = np.median(features_subset[predictions.astype(bool)], axis=0, keepdims=True).T
        distances, predictions, objective = evaluator.update('method_5', features_subset, mu, bag_to_index)
        # '''


    # generate plots

    for method in evaluator.get_methods():
        if len(evaluator.predictions[method]) == 0:
            continue
        
        objective = np.mean(evaluator.objectives[method])
        label = method.replace('_', ' ')
        label += f': {objective:.4f}'

        recall, prec = calc_prec_recall(evaluator.distances[method], y_true, evaluator.predictions[method])
        plt.plot(recall, prec, label=label)

        print(f'{label}')

    plot_prec_recall()
