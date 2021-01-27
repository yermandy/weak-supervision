import numpy as np
import matplotlib.pyplot as plt


def calc_prec_recall(y_true, distances):

    distances = np.array(distances)
    min_d = np.min(distances) + 1e-1
    max_d = np.max(distances)
    thresholds = np.linspace(min_d, max_d, min(100, len(distances)))

    recall, prec = [], []

    for th in thresholds:

        y_pred = np.where(distances <= th, 1, 0)

        tp = np.sum((y_true == y_pred) & (y_pred == 1))
        fp = np.sum((y_pred == 1) & (y_true != y_pred))
        fn = np.sum((y_pred == 0) & (y_true != y_pred))

        x = tp / (tp + fn + 1e-8)
        y = tp / (tp + fp + 1e-8)

        recall.append(x)
        prec.append(y)

    return recall, prec


def plot_prec_recall(recall, prec):
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.plot(recall, prec)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def filter_by_counts(metadata, min_counts=0, max_counts=np.inf):

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)

    _, indices, counts = np.unique(
        [paths, subjects], return_index=True, return_counts=True, axis=1
    )

    only_idx = (counts > min_counts) & (counts < max_counts)

    indices = indices[only_idx]
    counts = counts[only_idx]

    filtered = []

    for idx, cnt in zip(indices, counts):

        for i in range(cnt):
            filtered.append(idx + i)

    return filtered


if __name__ == "__main__":
    features = np.load("resources/features/ijbb_features.npy")

    metadata_file = "resources/ijbb_metadata.csv"
    metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",")

    indices = filter_by_counts(metadata, 0)

    metadata = metadata[indices]
    features = features[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    ground_true = metadata[:, 7].astype(int)

    tp = {"baseline_1": 0, "baseline_2": 0}
    fp = {"baseline_1": 0, "baseline_2": 0}
    fn = {"baseline_1": 0, "baseline_2": 0}

    # true_correct_matches = np.sum(ground_true == 1)

    distances = []
    y_true = []

    for subject in np.unique(subjects):

        idx = np.flatnonzero(subjects == subject)
        k = len(np.unique(paths[idx]))
        ones_k = np.ones(k)
        features_subset = features[idx]
        true_subset = ground_true[idx]

        # baseline 1: median

        median = np.median(features_subset, axis=0, keepdims=True).T
        cos_dist = (1 - features_subset @ median).flatten()
        arg_sorted = cos_dist.argsort()
        arg_sorted = arg_sorted[:k]
        true_k = true_subset[arg_sorted]
        tp["baseline_1"] += np.sum(true_k == 1)
        fp["baseline_1"] += np.sum((true_k == 1) & (ones_k != true_k))
        fn["baseline_1"] += np.sum((true_k == 0) & (ones_k != true_k))

        # TODO whole array or just [0] element?
        d = cos_dist[arg_sorted][0]
        distances.append(d)
        y = true_k[0]
        y_true.append(y)

        # baseline 2: two pass median

        median = np.median(features_subset[arg_sorted], axis=0, keepdims=True).T
        cos_dist = (1 - features_subset @ median).flatten()
        arg_sorted = cos_dist.argsort()
        arg_sorted = arg_sorted[:k]
        true_k = true_subset[arg_sorted]
        tp["baseline_2"] += np.sum(true_k == 1)
        fp["baseline_2"] += np.sum((true_k == 1) & (ones_k != true_k))
        fn["baseline_2"] += np.sum((true_k == 0) & (ones_k != true_k))

    recall, prec = calc_prec_recall(y_true, distances)

    plot_prec_recall(recall, prec)

    recall = tp["baseline_1"] / (tp["baseline_1"] + fn["baseline_1"] + 1e-8)
    precision = tp["baseline_1"] / (tp["baseline_1"] + fp["baseline_1"] + 1e-8)
    print("baseline 1")
    print(recall, precision)
    print()

    recall = tp["baseline_2"] / (tp["baseline_2"] + fn["baseline_2"] + 1e-8)
    precision = tp["baseline_2"] / (tp["baseline_2"] + fp["baseline_2"] + 1e-8)
    print("baseline 2")
    print(recall, precision)
