import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt


def milp(X, K, log=True, return_obj_val=False):

    model = gp.Model("model")

    if not log:
        model.Params.LogToConsole = 0

    # define constants

    # number of detections
    m = X.shape[0]
    
    # feature dimension
    d = X.shape[1]

    # partition of index array {1, ... , m} into bags I = [I_1, ... , I_k]
    _, K_counts = np.unique(K, return_counts=True)
    K = len(K_counts)
    I = []
    idx = 0
    for k in range(K):
        arr = []
        for j in range(K_counts[k]):
            arr.append(idx)
            idx += 1
        I.append(arr)
    K = len(K_counts)

    # median boundaries
    B = np.sum(X.max(axis=0) - X.min(axis=0))

    # median = np.median(X, axis=0)
    # print(median)

    # create variables

    xi_sum = gp.LinExpr()
    for k in range(K):
        xi = model.addVar(name=f"xi_{k}")
        xi_sum += xi

    for j in range(d):
        # model.addVar(name=f"mu_{j}", lb=-B, ub=B, obj=median[j])
        model.addVar(name=f"mu_{j}")

    for i in range(m):
        for j in range(d):
            # model.addVar(name=f"eta_{i}_{j}", obj=X[i, j] - median[j])
            model.addVar(name=f"eta_{i}_{j}")

    for i in range(m):
        model.addVar(vtype=gp.GRB.BINARY, name=f"alpha_{i}")

    # set objective

    model.setObjective(xi_sum, gp.GRB.MINIMIZE)
    model.update()

    # add constraints

    for i in range(m):
        for j in range(d):
            eta_i_j = model.getVarByName(f"eta_{i}_{j}")
            mu_j = model.getVarByName(f"mu_{j}")
            x_i_j = X[i, j]
            model.addConstr(eta_i_j >= mu_j - x_i_j, f"eta_{i}_{j} >= mu_{j} - x_{i}_{j}")
            model.addConstr(eta_i_j >= x_i_j - mu_j, f"eta_{i}_{j} >= x_{i}_{j} - mu_{j}")

    for k, I_k in enumerate(I):
        xi_k = model.getVarByName(f"xi_{k}")

        alpha_k_sum = gp.LinExpr()

        for i in I_k:
            eta_sum = gp.LinExpr()

            for j in range(d):
                eta_i_j = model.getVarByName(f"eta_{i}_{j}")
                eta_sum += eta_i_j

            alpha_i = model.getVarByName(name=f"alpha_{i}")
            model.addConstr(xi_k >= eta_sum - B * (1 - alpha_i), f"xi_{k} >= eta_{i}_sum - B * (1 - alpha_{i})")

            alpha_k_sum += alpha_i

        model.addLConstr(alpha_k_sum, gp.GRB.EQUAL, 1, name=f"alpha_{k}_sum = 1")

    # optimize model

    model.optimize()
    
    median = np.zeros(d)
    for j in range(d):
        median[j] = model.getVarByName(f"mu_{j}").x
    
    # print(model.objVal)
    # print(median)

    # display LP 
    # print(model.display())

    if return_obj_val:
        return median, model.objVal

    return median


def calc_obj(K, X, mu):
    objective = []
    _, unique_idx, unique_cnt = np.unique(K, return_index=True, return_counts=True)
    for i, c in zip(unique_idx, unique_cnt):
        d = mu - X[i:i+c].T
        objective.append(np.abs(d).sum(axis=0).min())
    return np.sum(objective)


if __name__ == "__main__":
    dataset = np.genfromtxt("synth/dataset_2.csv", skip_header=1, delimiter=",")

    X = dataset[:, 0]
    Y = dataset[:, 1]
    I = dataset[:, 2]
    K = dataset[:, 3]

    _, unique_idx, unique_cnt = np.unique(K, return_index=True, return_counts=True)
    iterator = list(zip(unique_idx, unique_cnt))

    colors = {1: "r", 2: "g", 3: "b"}

    markers = {0: "o", 1: "v"}

    for x, y, i, k in zip(X, Y, I, K):
        plt.scatter(x, y, color=colors[k], marker=markers[i])

    features = np.vstack((X, Y)).T
    median = np.median(features, axis=0, keepdims=True).T
    cos_dist = (1 - features @ median).flatten()
    median = median.flatten()

    plt.scatter(median[0], median[1], color='black', marker='x', label='median')

    median = milp(features, K)

    plt.scatter(median[0], median[1], color='magenta', marker='x', label='optimal')

    plt.legend()

    plt.show()