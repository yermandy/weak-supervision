import numpy as np
from src import *
from time import time


if __name__ == "__main__":
    features_training = "resources/features/imdb_features_training.npy"
    features_training = np.load(features_training)

    features_file = "resources/features/imdb_features.npy"
    features = np.load(features_file)

    metadata_training = "resources/imdb_metadata_not_annotated.csv"
    metadata_training = np.genfromtxt(metadata_training, dtype=str, delimiter=",", skip_header=1)

    metadata_file = "resources/imdb_metadata.csv"
    metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",", skip_header=1)


    subj_bag_indices_trining = parse_metadata(metadata_training)
    indices_training = filter_by_counts(subj_bag_indices_trining, 2)

    metadata_training = metadata_training[indices_training]
    features_training = features_training[indices_training]


    subj_bag_indices = parse_metadata(metadata)
    indices = filter_by_counts(subj_bag_indices, 2)

    metadata = metadata[indices]
    features = features[indices]

    paths_training = metadata_training[:, 0]
    subjects_training = metadata_training[:, 6].astype(int)
    labels_training = metadata_training[:, 7].astype(int)
    scores_training = metadata_training[:, 5].astype(float)

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    labels = metadata[:, 7].astype(int)

    y_true = []

    evaluator = Evaluator()

    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        idx_training = np.flatnonzero(subjects_training == subject)

        # scores_training_subset = scores_training[idx_training]
        # mask = scores_training_subset >= 0.5
        # idx_training = idx_training[mask]
        
        m = len(idx)
        
        if m == 1:
            continue
        
        # if m >= 30:
        #     continue

        paths_training_subset = paths_training[idx_training]
        features_training_subset = features_training[idx_training]
        labels_training_subset = labels_training[idx_training]

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = labels[idx]

        unique_paths_training, bag_to_index_training, index_to_bag_training = parse_paths(paths_training_subset)

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
        mu = median(features_training_subset)
        distances, predictions_method_1, objective = evaluator.update('_', features_training_subset, mu, bag_to_index_training)
        evaluator.update('median', features_subset, mu, bag_to_index)
        # '''

        #! Method 1: Median on annotated data
        '''
        mu = median(features_subset)
        distances, predictions_method_1, objective = evaluator.update('median from annotated', features_subset, mu, bag_to_index)
        # '''

        #! Method 2: Two-Stage Median
        # '''
        mu = median(features_training_subset[predictions_method_1.astype(bool)])
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
        # '''
        K = list(index_to_bag_training.values())

        mu, _ = suboptimal_median(features_training_subset, K)
        distances, predictions, objective = evaluator.update('no pca', features_subset, mu, bag_to_index)

        pca = PCA(min(len(features), 2))
        features_reduced = pca.fit_transform(features_subset)

        mu, _ = suboptimal_median(features_training_subset, K, 2)
        distances, predictions, objective = evaluator.update('pca {2}С 2D', features_reduced, mu, bag_to_index)

        mu, _ = adaptive_suboptimal_median(features_training_subset, K, bag_to_index_training, 2)
        distances, predictions, objective = evaluator.update('pca {2,4}С 2D', features_reduced, mu, bag_to_index)

        mu, _ = adaptive_suboptimal_median(features_training_subset, K, bag_to_index_training)
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
