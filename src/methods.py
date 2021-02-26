from sklearn.decomposition import PCA
import numpy as np
from milp import milp


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


def optimal_median(features, K, pca=True, n_components=2):
    """ Calculates optimal median using MILP across n_samples

    Parameters
    ----------
    features : np.array (n_samples, n_features)

    Returns
    -------
    mu: np.array (1, n_features)
    features: np.array (n_samples, n_features)
    """
    if pca:
        pca = PCA(min(len(features), n_components))
        features = pca.fit_transform(features)
    mu = milp(features, K, False)[np.newaxis, :].T
    return mu, features


def suboptimal_median(features, K, pca=True, n_components=2):
    """ Calculates suboptimal median across n_samples

    Parameters
    ----------
    features : np.array (n_samples, n_features)

    Returns
    -------
    mu: np.array (1, n_features)
    features: np.array (n_samples, min(n_samples, n_components))
    """

    if pca:
        pca = PCA(min(len(features), n_components))
        features = pca.fit_transform(features)

    lowest = np.inf
    for i, (f1, k1) in enumerate(zip(features, K)):
        
        distances_to = {}
        
        # from_id -> (dist, to_id)
        distances_to[k1] = (0, i)
        
        for j, (f2, k2) in enumerate(zip(features, K)):
            
            if i == j or k1 == k2:
                continue
                
            # d = f1 @ f2 / ( norm(f1) * norm(f2) )

            d = np.abs(f1 - f2).sum()

            if k2 in distances_to:
                if d < distances_to[k2][0]:
                    distances_to[k2] = (d, j)
            else:
                distances_to[k2] = (d, j)

        distances_to = np.array([*distances_to.values()])
        dists = distances_to[:, 0]
        dists_sum = np.sum(dists)
        
        if dists_sum < lowest:
            lowest = np.sum(dists)
            pred_idx = distances_to[:, 1]

    pred_idx.sort()

    mu = np.median(features[np.array(pred_idx, dtype=int)], axis=0, keepdims=True).T
    
    return mu, features