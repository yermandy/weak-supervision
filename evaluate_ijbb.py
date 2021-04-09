import numpy as np
from src import *
from time import time


if __name__ == "__main__":

    metadata_file = "resources/ijbb_metadata.csv"
    # features_file = "resources/features/ijbb_features.npy"
    features_file = "resources/features/ijbb_arcface_features.npy"

    paths, boxes, scores, subjects, labels, features = open_and_parse(metadata_file, features_file, 1)

    indices_img = np.flatnonzero(np.core.defchararray.find(paths , 'img') != -1)
    indices_frames = np.flatnonzero(np.core.defchararray.find(paths , 'frames') != -1)

    ijbb = np.genfromtxt('resources/ijbb_faces.csv', dtype=str, delimiter=",")
    ijbb_paths = ijbb[:, 0]
    ijbb_subjects = ijbb[:, 6].astype(int)

    ijbb_features_optimal = np.ones((len(ijbb), features.shape[1]))
    ijbb_features_median = np.ones((len(ijbb), features.shape[1]))
    ijbb_features_iterative_median = np.ones((len(ijbb), features.shape[1]))
    ijbb_features_bag_median = np.ones((len(ijbb), features.shape[1]))
    ijbb_features_bayes_median = np.ones((len(ijbb), features.shape[1]))

    bayes_distances = []
    bayes_predictions = []
    skipped_subjects = []

    eval_optimal_median = 1
    eval_median = 1
    eval_iterative_median = 1
    eval_bayes_median = 1
    eval_milp_median = 0
    eval_bag_median = 1

    train_only_on_images = 0
    test_only_on_images = 0
    test_only_on_frames = 0

    remove_single = 1
    save_features = 1
    pca_reduction = 0
    filter_by_quality = 0

    counter = 0

    evaluator = Evaluator()
    for s, subject in enumerate(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)
        idx_tst = idx.copy()
        idx_trn = idx.copy()

        if train_only_on_images:
            idx_trn = np.intersect1d(idx, indices_img)
        if test_only_on_frames:
            idx_tst = np.intersect1d(idx, indices_frames)
        if test_only_on_images:
            idx_tst = np.intersect1d(idx, indices_img)
        
        # m = len(idx_trn)
        # if m <= 1 or m >=50:
        #     continue

        if remove_single:
            _tmp = []
            bag_to_index = parse_paths(paths[idx_trn])[1]
            for k, v in bag_to_index.items():
                if len(v) == 1:
                    continue
                _tmp.extend(v)

            if len(_tmp) <= 1:
                skipped_subjects.append(subject)
                continue

            idx_trn = idx_trn[np.array(_tmp)]

        if filter_by_quality:
            scores_trn_subset = scores[idx_trn]
            mask = scores_trn_subset >= 0.0
            idx_trn = idx_trn[mask]

        scores_trn_subset = scores[idx_trn]
        paths_trn_subset = paths[idx_trn]
        features_trn_subset = features[idx_trn]
        labels_trn_subset = labels[idx_trn]

        labels_tst_subset = labels[idx_tst]
        paths_tst_subset = paths[idx_tst]
        features_tst_subset = features[idx_tst]

        unique_paths_trn, bag_to_index_trn, index_to_bag_trn = parse_paths(paths_trn_subset)
        unique_paths_tst, bag_to_index_tst, index_to_bag_tst = parse_paths(paths_tst_subset)        
        
        bags_tst_number = len(unique_paths_tst)
        bags_trn_number = len(unique_paths_trn)

        if min(bags_tst_number, bags_trn_number) <= 1:
            skipped_subjects.append(subject)
            continue

        counter += 1
        print(counter)

        evaluator.add_true(labels_tst_subset)

        if save_features:
            def add_features(predictions, ijbb_features):
                predictions = np.array(predictions, bool)
                paths_for_ijbb = paths_tst_subset[predictions]
                features_for_ijbb = features_tst_subset[predictions]

                for f, p in zip(features_for_ijbb, paths_for_ijbb):
                    ijbb_idx = np.flatnonzero((ijbb_paths == p) & (ijbb_subjects == subject))
                    ijbb_features[ijbb_idx] = f

        bag_trn = np.array(list(index_to_bag_trn.values()))
        bag_tst = np.array(list(index_to_bag_tst.values()))

        #! PCA reduction of features
        if pca_reduction:
            n_components = min(len(features_trn_subset), 2)
            pca = PCA(n_components)
            features_trn_subset = pca.fit_transform(features_trn_subset)

        #! Median from grouind true labels
        if eval_optimal_median:
            mu = median(features_trn_subset[labels_trn_subset.astype(bool)])
            distances, predictions, objective = evaluator.update('optimal', features_tst_subset, mu, bag_to_index_tst)
            if save_features:
                add_features(predictions, ijbb_features_optimal)
        
        #! Median
        if eval_median:
            mu = median(features_trn_subset)
            distances, predictions, objective = evaluator.update('median', features_tst_subset, mu, bag_to_index_tst)
            if save_features:
                add_features(predictions, ijbb_features_median)

        #! Iterative median
        if eval_iterative_median:
            preds = evaluator.update('_', features_trn_subset, mu, bag_to_index_trn)[1]
            mu = median(features_trn_subset)
            mu = iterative_median(features_trn_subset, mu, preds, bag_to_index_trn)
            distances, predictions, objective = evaluator.update('iterative', features_tst_subset, mu, bag_to_index_tst)
            if save_features:
                add_features(predictions, ijbb_features_iterative_median)

        #! Bayes median
        if eval_bayes_median:
            bag_trn = np.array(list(index_to_bag_trn.values()))
            bag_tst = np.array(list(index_to_bag_tst.values()))

            bag_tst = bag_tst + np.max(bag_trn) + 1

            trn_example = np.concatenate([np.ones(len(bag_trn)), np.zeros(len(bag_tst))])
            
            bag = np.concatenate([bag_trn, bag_tst])

            features_trn_tst = np.concatenate([features_trn_subset, features_tst_subset]).T

            _, S, P = bayes_learn(features_trn_tst, bag, trn_example=trn_example)

            S = 1 - S[len(bag_trn):]
            P_tst = P[len(bag_trn):]

            P_trn = P[:len(bag_trn)].astype(bool)
            
            mu = median(features_trn_subset[P_trn])
            
            _, predictions, _ = evaluator.update('bayes median', features_tst_subset, mu, bag_to_index_tst)
            
            if save_features:
                add_features(predictions, ijbb_features_bayes_median)

            bayes_distances.extend(S)
            bayes_predictions.extend(P_tst)

        #! Optimal MILP median
        if eval_milp_median:
            mu, features_reduced = optimal_median(features_trn_subset, list(index_to_bag_trn.values()))
            evaluator.update('milp median', features_reduced, mu, bag_to_index_tst)

        #! Bag median
        if eval_bag_median:
            mu = suboptimal_median(features_trn_subset, list(index_to_bag_trn.values()))
            distances, predictions, objective = evaluator.update('bag median', features_tst_subset, mu, bag_to_index_tst)
            if save_features:
                add_features(predictions, ijbb_features_bag_median)

    if eval_bayes_median:
        evaluator.distances['bayes median'] = bayes_distances
        evaluator.predictions['bayes median'] = bayes_predictions

    if save_features:
        np.save('skipped_subjects.npy', np.array(skipped_subjects))
        np.save('ijbb_features_optimal.npy', ijbb_features_optimal)
        np.save('ijbb_features_median.npy', ijbb_features_median)
        np.save('ijbb_features_bag_median.npy', ijbb_features_bag_median)
        np.save('ijbb_features_bayes_median.npy', ijbb_features_bayes_median)
        np.save('ijbb_features_iterative_median.npy', ijbb_features_iterative_median)

    evaluator.plot_prec_recall()