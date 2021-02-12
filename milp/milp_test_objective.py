import numpy as np
import matplotlib.pyplot as plt

from milp import milp, objective
from time import time

m = 10
d = 256

K = []
for i in range(m):
    K.append(i % 2)
K.sort()

bag_to_index = {}
for i, k in enumerate(K):
    if k not in bag_to_index:
        bag_to_index[k] = []
    bag_to_index[k].append(i)


for i in range(100):

    X = np.random.randint(-100, 100, (m, d))

    K = np.array(K)
    argsorted = np.argsort(K)

    K = K[argsorted]
    X = X[argsorted]

    mu, milp_obj_val = milp(X, K, False, True)
    mu = np.atleast_2d(mu).T

    milp_obj_val_1 = objective(bag_to_index, X, mu)

    mu = np.median(X, axis=0, keepdims=True).T

    obj = objective(bag_to_index, X, mu)

    print(milp_obj_val, '<' if milp_obj_val < obj else '>', obj)
