import numpy as np
import matplotlib.pyplot as plt

from milp import milp, calc_obj
from time import time


for i in range(100):

    X = np.random.randint(-100, 100, (10, 256))

    K = []
    for i in range(X.shape[0]):
        K.append(i % 2)

    K = np.array(K)
    argsorted = np.argsort(K)

    K = K[argsorted]
    X = X[argsorted]

    mu, milp_obj_val = milp(X, K, False, True)
    mu = np.atleast_2d(mu).T

    mu = np.median(X, axis=0, keepdims=True).T

    obj = calc_obj(K, X, mu)

    print(milp_obj_val, '<' if milp_obj_val < obj else '>', obj)
