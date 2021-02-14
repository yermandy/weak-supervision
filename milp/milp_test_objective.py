import numpy as np
import matplotlib.pyplot as plt

from milp import milp, objective
from time import time

m = 6
d = 2

K = []
for i in range(m):
    K.append(i % 4)
K.sort()
K = np.array(K)

bag_to_index = {}
for i, k in enumerate(K):
    if k not in bag_to_index:
        bag_to_index[k] = []
    bag_to_index[k].append(i)

colors = {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'c', 5: 'y'}

def predict(bag_to_index, dist, X, mu):
    y_pred = np.zeros(len(X))

    objective = []

    for indices in bag_to_index.values():
        argmin = dist[indices].argmin()
        pred_id = indices[argmin]
        y_pred[pred_id] = 1
        
        d = mu - X[indices].T
        objective.append(np.abs(d).sum(axis=0).min())
    
    objective = np.sum(objective)

    return y_pred, objective


for i in range(20):

    X = np.random.randn(m, d)
    X /= np.linalg.norm(X, axis=0)

    mu_median = np.median(X, axis=0, keepdims=True).T

    mu_normalized = mu_median / np.linalg.norm(mu_median)
    cos_dist = (1 - X @ mu_normalized).flatten()
    y_pred_method_1, obj_median = predict(bag_to_index, cos_dist, X, mu_median)

    two_stage_median = np.median(X[y_pred_method_1.astype(bool)], axis=0, keepdims=True).T

    obj_two_stage_median = objective(bag_to_index, X, two_stage_median)

    mu_milp, milp_obj_val = milp(X, K, True, True)
    mu_milp = np.atleast_2d(mu_milp).T

    milp_obj_val_1 = objective(bag_to_index, X, mu_milp)

    print(f'{obj_median:.4f} : {obj_two_stage_median:.4f} : {milp_obj_val:.4f}')

    # print(milp_obj_val, '<' if milp_obj_val < obj else '>', obj)
    
    if obj_median < milp_obj_val:

        for (x, y), k in zip(X, K):
            plt.scatter(x, y, color=colors[k])

        mu_median = mu_median.flatten()
        mu_milp = mu_milp.flatten()
        plt.scatter(mu_median[0], mu_median[1], color='black', marker='x', label='median')
        plt.scatter(mu_milp[0], mu_milp[1], color='magenta', marker='x', label='optimal')
        plt.legend()
        plt.show()
        break
