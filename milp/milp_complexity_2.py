import numpy as np
import matplotlib.pyplot as plt

from milp import milp
from time import time

# feature dim
d = 16

# detections
m = 20

# 
Ms = [m]

# bags from 1 to m
Ks = np.arange(1, m + 1 , dtype=int)

T1s, T2s = [], []

for j in range(3):
# for d in Ds:

    # features
    X = np.random.randn(m, d)

    # time
    T1, T2 = [], []

    for K in Ks:

        K_arr = []
        for i in range(m):
            K_arr.append(i % K)
        K_arr.sort()

        t1 = time()
        milp(X, K_arr, False)
        t2 = time()
        T1.append(t2 - t1)
        print(K, t2 - t1)

        # t1 = time()
        # milp(X, K_arr, False)
        # t2 = time()
        # T2.append(t2 - t1)
        # print(K, t2 - t1)

    print()

    T1s.append(T1)
    # T2s.append(T2)


T1s = np.mean(T1s, axis=0)
# T2s = np.mean(T2s, axis=0)

plt.plot(Ks, T1s, label=f'bounded')
# plt.plot(Ks, T2s, label=f'unbounded')

print()

plt.xlabel('K')
plt.ylabel('t(K) [s]')
plt.tight_layout()
plt.legend()
# plt.savefig(f'results/milp/milp_m({m}).png', dpi=300)
plt.show()