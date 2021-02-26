import numpy as np
from src import *
from time import time


if __name__ == "__main__":
    # features_file = "resources/features/ijbb_features.npy"
    features_file = "resources/features/imdb_features.npy"
    features = np.load(features_file)

    # metadata_file = "resources/ijbb_metadata.csv"
    metadata_file = "resources/imdb_metadata.csv"
    metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",", skip_header=1)

    subj_bag_indices = parse_metadata(metadata)

    indices = filter_by_counts(subj_bag_indices, 2)

    metadata = metadata[indices]
    features = features[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    labels = metadata[:, 7].astype(int)

    y_true = []

    evaluator = Evaluator()

    counter = 0
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        
        m = len(idx)
        
        if m == 1:
            continue
        
        # if m >= 30:
        #     continue

        counter += 1
        print(counter)

        paths_subset = paths[idx]
        features_subset = features[idx]
        labels_subset = labels[idx]

        if len(np.unique(labels_subset)) == 1:
            continue

        y_true.extend(labels_subset)

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
        bags_number = len(unique_paths)

        #! PCA reduction
        ''' 
        n_components = min(len(features_subset), 2)
        pca = PCA(n_components)
        features_subset = pca.fit_transform(features_subset)
        # '''

        #! Method *: Best possible median
        # '''
        mu = median(features_subset[labels_subset.astype(bool)])
        evaluator.update('method_*', features_subset, mu, bag_to_index)
        # '''
        
        #! Method 1: Median
        # '''
        mu = median(features_subset)
        distances, predictions_method_1, objective = evaluator.update('method_1', features_subset, mu, bag_to_index)
        # '''

        #! Method 2: Two-Stage Median
        # '''
        mu = median(features_subset[predictions_method_1.astype(bool)])
        evaluator.update('method_2', features_subset, mu, bag_to_index)
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
        mu, features_reduced = suboptimal_median(features_subset, list(index_to_bag.values()))
        distances, predictions, objective = evaluator.update('method_5', features_reduced, mu, bag_to_index)
        
        # mu = np.median(features_subset[predictions.astype(bool)], axis=0, keepdims=True).T
        # distances, predictions, objective = evaluator.update('method_5', features_subset, mu, bag_to_index)
        # '''

    #! Generate plots

    for method in evaluator.get_methods():
        if len(evaluator.predictions[method]) == 0:
            continue
        
        objective = np.mean(evaluator.objectives[method])
        label = method.replace('_', ' ')
        label += f': {objective:.4f}'

        recall, prec = calc_prec_recall(evaluator.distances[method], y_true, evaluator.predictions[method])
        plt.plot(recall, prec, label=label)

        print(f'{label}')

    plot_prec_recall()
