import numpy as np
import matplotlib.pyplot as plt

from milp import milp
from time import time

# feature dim
Ds = [8, 16, 32, 64]

# detections
m = 30

# 
Ms = [m]

# bags from 1 to m
Ks = np.arange(1, m + 1 , dtype=int)

# for i in range(10):
for d in Ds:

    # features
    X = np.random.randn(m, d)

    # time
    T = []

    for K in Ks:

        K_arr = []
        for i in range(m):
            K_arr.append(i % K)
        K_arr.sort()

        t1 = time()
        milp(X, K_arr, False)
        t2 = time()
        T.append(t2 - t1)
        
        print(K, t2 - t1)

    plt.plot(Ks, T, label=f'd = {d}')

    print()

plt.xlabel('K')
plt.ylabel('t(K) [s]')
plt.tight_layout()
plt.legend()
plt.savefig(f'results/milp/milp_m({m}).png', dpi=300)
# plt.show()