import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import *
from tqdm import tqdm
from easydict import EasyDict as edict


sns.set_style('white')
plt.style.use('seaborn-deep')


experiment_dataset = 'imdb_train'
experiment_dataset = 'imdb_test'
experiment_path = f'experiments/visualize_stats'

# os.makedirs(f'{root}/{experiment_path}', exist_ok=True)

def open_and_parse(metadata):
    metadata = open_metadata(metadata)

    subj_bag_indices = parse_metadata(metadata)
    indices = filter_by_counts(subj_bag_indices, 2)

    metadata = metadata[indices]

    paths = metadata[:, 0]
    subjects = metadata[:, 6].astype(int)
    labels = metadata[:, 7].astype(int)
    scores = metadata[:, 5].astype(float)

    return paths, subjects, labels, scores


def get_hist(metadata):
    paths, subjects, _, _ = open_and_parse(metadata)
    hist = edict()
    
    img_per_subject = []
    dets_per_image = []
    dets_per_subject = []

    for subject in tqdm(np.unique(subjects)):

        idx = np.flatnonzero(subjects == subject)

        paths_subset = paths[idx]

        unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)

        bags_number = len(unique_paths)

        dets_per_image_in_subject = [len(indices) for indices in bag_to_index.values()]
        dets_per_image.extend(dets_per_image_in_subject)

        dets_per_subject.append(np.sum(dets_per_image_in_subject))

        img_per_subject.append(bags_number)

    hist.img_per_subject = np.array(img_per_subject)
    hist.dets_per_image = np.array(dets_per_image)
    hist.dets_per_subject = np.array(dets_per_subject)
    return hist


ijbb = get_hist(f"resources/ijbb_metadata.csv")
imdb = get_hist(f"resources/imdb_train_metadata.csv")

#! Plot images per subject histogram

max_bags_per_subject = 200
ijbb.img_per_subject = ijbb.img_per_subject[ijbb.img_per_subject <= max_bags_per_subject]
imdb.img_per_subject = imdb.img_per_subject[imdb.img_per_subject <= max_bags_per_subject]

bins = 200
plt.hist(ijbb.img_per_subject, bins=bins, label='ijbb', alpha=0.5, density=True, histtype='stepfilled')
plt.hist(imdb.img_per_subject, bins=bins, label='imdb', alpha=0.5, density=True, histtype='stepfilled')

plt.xlim((0, 100))
plt.title('images per subject')
plt.tight_layout()
plt.legend()
plt.savefig(f'{experiment_path}/images_per_subject.png', dpi=300)
# plt.show()
plt.close()

#! Plot detections per image histogram

max_dets_per_image = 200
ijbb.dets_per_image = ijbb.dets_per_image[ijbb.dets_per_image <= max_dets_per_image]
imdb.dets_per_image = imdb.dets_per_image[imdb.dets_per_image <= max_dets_per_image]

bins = 200
plt.hist(ijbb.dets_per_image, bins=bins, label='ijbb', alpha=0.5, density=True, histtype='stepfilled')
plt.hist(imdb.dets_per_image, bins=bins, label='imdb', alpha=0.5, density=True, histtype='stepfilled')

plt.xlim((0, 30))
plt.title('detections per image')
plt.tight_layout()
plt.legend()
plt.savefig(f'{experiment_path}/detections_per_image.png', dpi=300)
# plt.show()
plt.close()

#! Plot detections per subject histogram

max_dets_per_subject = 1000
ijbb.dets_per_subject = ijbb.dets_per_subject[ijbb.dets_per_subject <= max_dets_per_subject]
imdb.dets_per_subject = imdb.dets_per_subject[imdb.dets_per_subject <= max_dets_per_subject]

bins = 200
plt.hist(ijbb.dets_per_subject, bins=bins, label='ijbb', alpha=0.5, density=True, histtype='stepfilled')
plt.hist(imdb.dets_per_subject, bins=bins, label='imdb', alpha=0.5, density=True, histtype='stepfilled')

plt.xlim((0, 500))
plt.title('detections per subject')
plt.tight_layout()
plt.legend()
plt.savefig(f'{experiment_path}/detections_per_subject.png', dpi=300)
# plt.show()
plt.close()
