import numpy as np
from typing import List
from .metrics import *
from .plots import *

class Evaluator:
    def __init__(self, methods: List[str] = None) -> None:
        self.distances = {}
        self.predictions = {}
        self.objectives = {}
        self.true = []

        if methods is not None:
            for method in methods:
                self.add_method(method)

    def get_methods(self):
        return list(self.distances.keys())

    def add_true(self, labels_to_add: list):
        self.true.extend(labels_to_add)

    def add_method(self, method: str):
        self.distances[method] = []
        self.predictions[method] = []
        self.objectives[method] = []

    def update(self, method: str, features: np.array, mu: np.array, bag_to_index: dict):
        if method not in self.distances:
            self.add_method(method)

        mu_normalized = mu / np.linalg.norm(mu)
        # distances = np.linalg.norm(features - mu_normalized.flatten(), axis=1)
        distances = np.linalg.norm(features - mu_normalized.flatten(), ord=1, axis=1)
        # distances = (1 - features @ mu_normalized).flatten()

        predictions, objective = predict(bag_to_index, distances, features, mu)

        self.distances[method].extend(distances)
        self.predictions[method].extend(predictions)
        self.objectives[method].append(objective)

        return distances, predictions, objective

    def plot_prec_recall(self):
        for method in self.get_methods():
            if len(self.distances[method]) == 0 or method == '_':
                continue
            
            # objective = np.mean(evaluator.objectives[method])
            label = method.replace('_', ' ')
            # label += f': {objective:.2f}'

            recall, prec = calc_prec_recall(self.distances[method], self.true, self.predictions[method])
            plt.plot(recall, prec, label=label)

            print(f'{label}')

        plot_prec_recall()


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
