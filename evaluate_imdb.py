import numpy as np
from src import *
from time import time


if __name__ == "__main__":
    metadata_trn = "resources/imdb_train_metadata.csv"
    features_trn = "resources/features/imdb_train_features.npy"

    paths_trn, subjects_trn, labels_trn, scores_trn, features_trn = open_and_parse(metadata_trn, features_trn)

    metadata_tst = "resources/imdb_test_metadata.csv"
    features_tst = "resources/features/imdb_test_features.npy"

    paths_tst, subjects_tst, labels_tst, scores_tst, features_tst = open_and_parse(metadata_tst, features_tst)

    y_true = []

    evaluator = Evaluator()

    counter = 0
    for s, subject in enumerate(np.unique(subjects_tst)):

        idx = np.flatnonzero(subjects_tst == subject)
        idx_training = np.flatnonzero(subjects_trn == subject)

        # scores_trn_subset = scores_trn[idx_training]
        # mask = scores_trn_subset >= 0.5
        # idx_training = idx_training[mask]
        
        m = len(idx)
        
        if m == 1:
            continue
        
        # if m >= 30:
        #     continue

        paths_trn_subset = paths_trn[idx_training]
        features_trn_subset = features_trn[idx_training]
        labels_trn_subset = labels_trn[idx_training]

        paths_subset = paths_tst[idx]
        features_subset = features_tst[idx]
        labels_subset = labels_tst[idx]

        unique_paths_trn, bag_to_index_training, index_to_bag_training = parse_paths(paths_trn_subset)

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        bags_number = len(unique_paths)

        # if bags_number == 1:
        #     continue

        counter += 1
        print(counter)

        y_true.extend(labels_subset)

        #! PCA reduction
        ''' 
        n_components = min(len(features_subset), 2)
        pca = PCA(n_components)
        features_subset = pca.fit_transform(features_subset)
        # '''

        #! Method *: Best possible median
        # '''
        mu = median(features_subset[labels_subset.astype(bool)])
        evaluator.update('optimal', features_subset, mu, bag_to_index)
        # '''
        
        #! Method 1: Median on training data
        # '''
        mu = median(features_trn_subset)
        distances, predictions_method_1, objective = evaluator.update('_', features_trn_subset, mu, bag_to_index_training)
        evaluator.update('median', features_subset, mu, bag_to_index)
        # '''

        #! Method 1: Median on annotated data
        '''
        mu = median(features_subset)
        distances, predictions_method_1, objective = evaluator.update('median from annotated', features_subset, mu, bag_to_index)
        # '''

        #! Method 2: Two-Stage Median
        # '''
        mu = median(features_trn_subset[predictions_method_1.astype(bool)])
        evaluator.update('two-stage', features_subset, mu, bag_to_index)
        # '''

        #! Method 3: Average of Medians
        '''
        mu = mean(features_subset[predictions_method_1.astype(bool)])
        evaluator.update('method_3', features_subset, mu, bag_to_index)
        # '''

        #! Method 4: Optimal median
        '''
        mu, features_reduced = optimal_median(features_subset, list(index_to_bag.values()))
        distances, predictions, objective = evaluator.update('method_4', features_reduced, mu, bag_to_index)
        # '''

        #! Method 5: Suboptimal median
        '''
        K = list(index_to_bag_training.values())

        mu, _ = suboptimal_median(features_trn_subset, K)
        distances, predictions, objective = evaluator.update('no pca', features_subset, mu, bag_to_index)

        pca = PCA(min(len(features_tst), 2))
        features_reduced = pca.fit_transform(features_subset)

        mu, _ = suboptimal_median(features_trn_subset, K, 2)
        distances, predictions, objective = evaluator.update('pca {2}С 2D', features_reduced, mu, bag_to_index)

        mu, _ = adaptive_suboptimal_median(features_trn_subset, K, bag_to_index_training, 2)
        distances, predictions, objective = evaluator.update('pca {2,4}С 2D', features_reduced, mu, bag_to_index)

        mu, _ = adaptive_suboptimal_median(features_trn_subset, K, bag_to_index_training)
        distances, predictions, objective = evaluator.update('pca {2,4}C 256D', features_subset, mu, bag_to_index)
        # '''

    #! Generate plots

    for method in evaluator.get_methods():

        if len(evaluator.predictions[method]) == 0 or method == '_':
            continue
        
        objective = np.mean(evaluator.objectives[method])
        label = method.replace('_', ' ')
        label += f': {objective:.4f}'

        recall, prec = calc_prec_recall(evaluator.distances[method], y_true, evaluator.predictions[method])
        plt.plot(recall, prec, label=label)

        print(f'{label}')

    plot_prec_recall()
