import numpy as np
from src import *
from time import time


if __name__ == "__main__":
    features_file = "resources/features/ijbb_features.npy"
    metadata_file = "resources/ijbb_metadata.csv"

    paths, subjects, labels, scores, features = open_and_parse(metadata_file, features_file)

    evaluator = Evaluator()

    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)

        scores_subset = scores[idx]
        mask = scores_subset >= 0.0
        idx = idx[mask]

        m = len(idx)
        
        if m <= 1:
            continue
        
        # if m >= 30:
        #     continue

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = labels[idx]

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        bags_number = len(unique_paths)

        if bags_number == 1:
            continue

        counter += 1
        print(counter)

        evaluator.add_true(labels_subset)

        #! PCA reduction of features
        ''' 
        n_components = min(len(features_subset), 2)
        pca = PCA(n_components)
        features_subset = pca.fit_transform(features_subset)
        # '''

        #! Best possible median
        # '''
        mu = median(features_subset[labels_subset.astype(bool)])
        evaluator.update('optimal', features_subset, mu, bag_to_index)
        # '''
        
        #! Median
        # '''
        mu = median(features_subset)
        distances, predictions_method_1, objective = evaluator.update('median', features_subset, mu, bag_to_index)
        # '''

        #! Two-Stage Median
        # '''
        mu = median(features_subset[predictions_method_1.astype(bool)])
        evaluator.update('two-pass', features_subset, mu, bag_to_index)
        # '''

        #! Average of Medians
        '''
        mu = mean(features_subset[predictions_method_1.astype(bool)])
        evaluator.update('method_3', features_subset, mu, bag_to_index)
        # '''

        #! Optimal median
        '''
        mu, features_reduced = optimal_median(features_subset, list(index_to_bag.values()))
        distances, predictions, objective = evaluator.update('method_4', features_reduced, mu, bag_to_index)
        # '''

        #! Suboptimal median without PCA
        # '''
        # mu, features_reduced = suboptimal_median(features_subset, list(index_to_bag.values()))
        # distances, predictions, objective = evaluator.update('no pca', features_subset, mu, bag_to_index)

        #! Suboptimal median with normalized features
        # '''
        # features_normalized = (features_subset - features_subset.mean(axis=0)) / features_subset.var(axis=0)
        # mu, features_reduced = suboptimal_median(features_normalized, list(index_to_bag.values()))
        # distances, predictions, objective = evaluator.update('normalized', features_normalized, mu, bag_to_index)
        # '''

        #! Different quality thresholds
        '''
        mu, features_reduced = suboptimal_median(features_subset, list(index_to_bag.values()))
        distances, predictions, objective = evaluator.update('th 0', features_subset, mu, bag_to_index)
        
        threshold = 0.25
        mask = scores_subset >= threshold
        features_thresholded = features_subset[mask]
        K = np.array(list(index_to_bag.values()))
        K = K[mask]
        mu, features_reduced = suboptimal_median(features_thresholded, K)
        distances, predictions, objective = evaluator.update('th 0.25', features_subset, mu, bag_to_index)

        threshold = 0.5
        mask = scores_subset >= threshold
        features_thresholded = features_subset[mask]
        K = np.array(list(index_to_bag.values()))
        K = K[mask]
        mu, features_reduced = suboptimal_median(features_thresholded, K)
        distances, predictions, objective = evaluator.update('th 0.5', features_subset, mu, bag_to_index)

        threshold = 0.75
        mask = scores_subset >= threshold
        features_thresholded = features_subset[mask]
        K = np.array(list(index_to_bag.values()))
        K = K[mask]
        mu, features_reduced = suboptimal_median(features_thresholded, K)
        distances, predictions, objective = evaluator.update('th 0.75', features_subset, mu, bag_to_index)

        threshold = 0.95
        mask = scores_subset >= threshold
        features_thresholded = features_subset[mask]
        K = np.array(list(index_to_bag.values()))
        K = K[mask]
        mu, features_reduced = suboptimal_median(features_thresholded, K)
        distances, predictions, objective = evaluator.update('th 0.95', features_subset, mu, bag_to_index)
        # '''

        #! Different PCA evaluation
        '''
        mu, features_reduced = suboptimal_median(features_subset, list(index_to_bag.values()), 2)
        distances, predictions, objective = evaluator.update('pca {2}С 2D', features_reduced, mu, bag_to_index)

        mu, features_reduced = adaptive_suboptimal_median(features_subset, list(index_to_bag.values()), bag_to_index, 2)
        distances, predictions, objective = evaluator.update('pca {2,4}C 2D', features_reduced, mu, bag_to_index)

        mu, features_reduced = adaptive_suboptimal_median(features_subset, list(index_to_bag.values()), bag_to_index)
        distances, predictions, objective = evaluator.update('pca {2,4}C 256D', features_reduced, mu, bag_to_index)
        # '''

    #! Generate plots

    for method in evaluator.get_methods():
        if len(evaluator.predictions[method]) == 0:
            continue
        
        objective = np.mean(evaluator.objectives[method])
        label = method.replace('_', ' ')
        label += f': {objective:.2f}'

        recall, prec = calc_prec_recall(evaluator.distances[method], evaluator.true, evaluator.predictions[method])
        plt.plot(recall, prec, label=label)

        print(f'{label}')

    plot_prec_recall()
