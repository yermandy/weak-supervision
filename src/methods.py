from numpy.linalg import norm
from sklearn.decomposition import PCA
import numpy as np
from milp import milp
import itertools


def median(features: np.array) -> np.array:
    """ Calculates median across n_samples

    Parameters
    ----------
    features : np.array (n_samples, n_features)

    Returns
    -------
    mu: np.array (1, n_features)
    """
    return np.median(features, axis=0, keepdims=True).T


def mean(features: np.array) -> np.array:
    """ Calculates mean across n_samples

    Parameters
    ----------
    features : np.array (n_samples, n_features)

    Returns
    -------
    mu: np.array (1, n_features)
    """
    return np.mean(features, axis=0, keepdims=True).T


def optimal_median(features: np.array, bags: np.array, pca=True, n_components=2):
    """ Calculates optimal median using MILP across n_samples

    Parameters
    ----------
    features : np.array (n_samples, n_features)
    K: np.array (n_samples)

    Returns
    -------
    mu: np.array (1, n_features)
    features: np.array (n_samples, n_features)
    """
    if pca:
        pca = PCA(min(len(features), n_components))
        features = pca.fit_transform(features)
    mu = milp(features, bags, False)[np.newaxis, :].T
    return mu, features


def suboptimal_median(features, bags, n_components=None, return_feat=False, return_pred=False):
    """ Calculates suboptimal median across n_samples.
        Applies PCA to features if n_components is not None

    Parameters
    ----------
    features : np.array (n_samples, n_features)
    K: np.array (n_samples)
    n_components: None or int (n_components)
    return_pred: bool

    Returns
    -------
    mu: np.array (1, n_features)
    features: np.array (n_samples, min(n_samples, n_components))
    """

    if n_components is not None:
        pca = PCA(min(len(features), n_components))
        features = pca.fit_transform(features)
        features /= norm(features, keepdims=True, axis=1)

    lowest = np.inf

    for i, (f1, k1) in enumerate(zip(features, bags)):
        
        # from_id -> (dist, to_id)
        distances_to = {k1: (0, i)}

        distances = np.abs(f1 - features).sum(1)
        
        for j, k2 in enumerate(bags):
            
            if k1 == k2:
                continue
                        
            d = distances[j]

            if k2 in distances_to:
                if d < distances_to[k2][0]:
                    distances_to[k2] = (d, j)
            else:
                distances_to[k2] = (d, j)

        distances_to = np.array([*distances_to.values()])
        dists_sum = distances_to[:, 0].sum()
        
        if dists_sum < lowest:
            lowest = dists_sum
            pred_idx = distances_to[:, 1]

    pred_idx = pred_idx.astype(int)
    mu = median(features[pred_idx])

    to_return = [mu]
    
    if return_feat:
        to_return += [features]
        
    if return_pred:
        to_return += [pred_idx]
    
    return to_return[0] if len(to_return) == 1 else to_return


def adaptive_suboptimal_median(features, bags, bag_to_index, n_components=None):

    # components = np.linspace(0, 256, 3, dtype=int)
    components = [2, 4]

    predictions = np.zeros(len(bags))

    for c in components:
        _, pred_idx = suboptimal_median(features, bags, c, return_pred=True)
        for p in pred_idx:
            predictions[p] += 1
    
    final_predictions = []
    for indices in bag_to_index.values(): 
        counters = predictions[indices]
        p = np.argmax(counters)
        final_predictions.append(indices[p])

    if n_components is not None:
        pca = PCA(min(len(features), n_components))
        features = pca.fit_transform(features)

    final_predictions = np.array(final_predictions, dtype=int)
    mu = median(features[final_predictions])

    return mu, features


def iterative_median(features, mu, alpha, bag_to_index):
    mu /= np.linalg.norm(mu)
    
    M = len(alpha)
    alpha_new = np.zeros_like(alpha)
    bag_to_index = bag_to_index.values()
    alpha_old = alpha

    while not np.array_equal(alpha_old, alpha_new):
        alpha_old = alpha_new
        
        # new alpha
        distances = np.abs(features - mu.flatten()).sum(1)
        # print(distances.shape)
        min_obj = np.inf
        obj = 0
        pred = []

        for idx in bag_to_index:
            dist = distances[idx]
            k = np.argmin(dist)
            obj += dist[k]
            pred.append(idx[k])

        if obj < min_obj:
            min_obj = obj
            pred_target = pred

        alpha_new = np.zeros(M)
        alpha_new[pred_target] = 1

        # print(obj)

        # new mu
        mu = median(features[pred_target])
        mu /= np.linalg.norm(mu)

    return mu


def franc_fast(features, bag_to_index):
    min_obj = np.inf
    bag_to_index = bag_to_index.values()

    for feature in features:
        distances = np.abs(features - feature).sum(1)
        obj = 0
        pred = []

        for idx in bag_to_index:
            dist = distances[idx]
            k = np.argmin(dist)
            obj += dist[k]
            pred.append(idx[k])

        if obj < min_obj:
            min_obj = obj
            pred_target = pred

    return pred_target


from scipy.special import logsumexp


def bayes_learn(features, bag, trn_example=[], sigma=1):
    bag = np.array(bag)
    n_faces = len(bag)
    n_bags = max(bag) + 1
    n_dims = features.shape[0]

    if len(trn_example) == 0:
        trn_example = np.ones(n_faces)
    
    logD = np.zeros([n_faces, n_bags])
    for i in range(n_faces):
        mu = features[:,i].reshape([n_dims,1])        
        dist = np.sum(np.absolute(features - mu), axis=0)

        for b in range(n_bags):
            idx = np.flatnonzero(bag == b)
            # take only bags with all features marked as training
            if np.all(trn_example[idx] == 1):
                logD[i,b] = logsumexp( -dist[idx] / sigma )

    sumLogD = np.sum(logD, axis=1)
    #print(np.where(np.isnan(sumLogD)))
        
    preds = []
    P = np.zeros(n_faces)
    for b in range(n_bags):

        idx = np.flatnonzero(bag == b)
        weight = sumLogD - logD[:, b]

        tmp = np.zeros([n_faces, len(idx)])
        cnt = 0
        for i in idx:
            mu = features[:,i].reshape([n_dims,1])
            dist = np.sum( np.absolute(features-mu),axis=0)
            tmp[:,cnt] = -dist/sigma + weight
            cnt = cnt + 1

        K = np.max(tmp)
        tmp = tmp - K
        cnt = 0
        for i in idx:
            P[i] = np.sum( np.exp(tmp[:,cnt]))
            cnt = cnt + 1

        # nconst = np.sum( P[idx])
        P[idx] = P[idx] / np.sum(P[idx])

        pred_idx = np.argmax(P[idx])
        
        preds.append(idx[pred_idx])

    preds = np.array(preds)
    
    mu = median(features[:, preds].T)
    # print(mu.shape)

    predictions = np.zeros(n_faces)
    predictions[preds] = 1
    return mu, P, predictions
