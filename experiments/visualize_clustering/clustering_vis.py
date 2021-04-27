import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from src import *

from sklearn.decomposition import PCA
from PIL import Image, ImageDraw


experiment_path = 'experiments/visualize_clustering'
experiment_dataset = 'ijbb'
subject = 163

# features = np.load(f"resources/features/{experiment_dataset}_features.npy")
features = f"resources/features/{experiment_dataset}_arcface_features.npy"

metadata_file = f"resources/{experiment_dataset}_metadata.csv"

paths, boxes, scores, subjects, labels, features = open_and_parse(metadata_file, features, 2)

idx = np.flatnonzero(subjects == subject)

# idx_img = np.flatnonzero(np.core.defchararray.find(paths , 'img') != -1)
idx_frames = np.flatnonzero(np.core.defchararray.find(paths , 'img') == -1)

idx = np.intersect1d(idx, idx_frames)

print('subject:', subject)
print('detections:', len(idx))

paths_subset = paths[idx]
boxes_subset = boxes[idx]
features_subset = features[idx]
labels_subset = labels[idx]

unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)

evaluator = Evaluator()

K = list(index_to_bag.values())
medoid = suboptimal_median(features_subset, K)
distances, predictions, objective = evaluator.update('_', features_subset, medoid, bag_to_index)


mu_ground_true = median(features_subset[labels_subset.astype(bool)]).T

concatenated = np.concatenate((features_subset, medoid.T, mu_ground_true), axis=0)

pca = PCA(3)
concatenated = pca.fit_transform(concatenated)
concatenated /= norm(concatenated, keepdims=True, axis=1)

features_subset = concatenated[:-2]
medoid = concatenated[-2]
mu_ground_true = concatenated[-1]

markers = {0: "o", 1: "v"}

unique_K = len(unique_paths)

ax = plt.figure().add_subplot(projection='3d')
ax.set_box_aspect([1, 1, 1])

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 0.985
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
# ax.plot_surface(x, y, z, color='black', alpha=0.05)
ax.plot_surface(x, y, z, color='white', alpha=0.25)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)


for (x, y, z), p, k in zip(features_subset, labels_subset, K):
    color = cm.gist_rainbow((k + 1) / unique_K)
    # plt.scatter(x, y, marker=markers[p], color=color)
    ax.scatter3D(x, y, z, marker=markers[p], color=color)

# Plot medoid
medoid = medoid.flatten()
medoid = medoid / norm(medoid)
# plt.scatter(mu[0], mu[1], marker='+', color='magenta', s=100)
ax.scatter3D(medoid[0], medoid[1], medoid[2], marker='+', color='magenta', s=100)
ax.quiver(0, 0, 0, medoid[0], medoid[1], medoid[2], color='magenta', arrow_length_ratio=0.1)

# Plot ground true
medoid = np.median(features_subset[labels_subset.astype(bool)], axis=0)
medoid = medoid / norm(medoid)
# plt.scatter(mu[0], mu[1], marker='x', color='black', s=70)
ax.scatter3D(medoid[0], medoid[1], medoid[2], marker='x', color='blue', s=70)
ax.quiver(0, 0, 0, medoid[0], medoid[1], medoid[2], color='blue', arrow_length_ratio=0.1)

# Plot [0, 0, 0]
ax.scatter(0, 0, 0, color='black')

plt.show()