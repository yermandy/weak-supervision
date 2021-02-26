import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dominate
import torchvision.transforms as transforms

from dominate.tags import *
from src.evaluator import *
from src.parse import *
from milp import milp

from sklearn.decomposition import PCA
from PIL import Image, ImageDraw



experiment_path = 'experiments/visualize_clustering'


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])


def crop(image, bb, scale=0.5):
    x1, y1, x2, y2 = bb
    w_scale = ((x2 - x1) * scale) / 2
    h_scale = ((y2 - y1) * scale) / 2
    x1 -= int(w_scale)
    y1 -= int(h_scale)
    x2 += int(w_scale)
    y2 += int(h_scale)
    return image.crop((x1, y1, x2, y2))


def get_image_name(path, idx):
    return path.split('/')[1].split('.')[0] + f"_{idx}.png"


features = np.load("resources/features/ijbb_features.npy")

metadata_file = "resources/ijbb_metadata.csv"
metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",", skip_header=1)

subj_bag_indices = parse_metadata(metadata)

indices = filter_by_counts(subj_bag_indices, 2)

metadata = metadata[indices]
features = features[indices]


paths = metadata[:, 0]
boxes = metadata[:, 1:5].astype(int)
subjects = metadata[:, 6].astype(int)
labels = metadata[:, 7].astype(int)


subject = 11802
idx = np.flatnonzero(subjects == subject)


paths_subset = paths[idx]
boxes_subset = boxes[idx]
features_subset = features[idx]
labels_subset = labels[idx]


unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)


evaluator = Evaluator()


n_components = 2
pca = PCA(n_components)
features_reduced = pca.fit_transform(features_subset)
K = list(index_to_bag.values())
mu, alphas = milp(features_reduced, K, False, return_alphas=True)
mu = np.atleast_2d(mu).T
distances, predictions, objective = evaluator.update('method_4', features_reduced, mu, bag_to_index)


markers = {0: "o", 1: "v"}
# colors = {}

unique_K = len(unique_paths)

for (x, y), p, k in zip(features_reduced, predictions, K):
    color = cm.gist_rainbow((k + 1) / unique_K)
    plt.scatter(x, y, marker=markers[p], color=color)

mu = mu.flatten()
plt.scatter(mu[0], mu[1], marker='x', color='magenta')

mu = np.median(features_reduced[labels_subset.astype(bool)], axis=0)
plt.scatter(mu[0], mu[1], marker='x', color='black')

plt.show()