import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import dominate
import torchvision.transforms as transforms

from dominate.tags import *
from evaluate import *
from PIL import Image, ImageDraw


experiment_path = 'experiments/dist_to_median_visualization'


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

def render_faces(distances, paths_subset, boxes_subset):
    argsorted = np.argsort(distances)
    
    paths_subset = paths_subset[argsorted]
    boxes_subset = boxes_subset[argsorted]
    # labels_subset = labels_subset[argsorted]
    method_div = div(style='width: max-content;')

    for box, path, idx in zip(boxes_subset, paths_subset, argsorted):
        image_name = get_image_name(path, idx)

        img_div = img(src=f'../../images/single/{image_name}', style='width: 150px;')
        method_div.add(img_div)
    
    return method_div


features = np.load("resources/features/ijbb_features.npy")

metadata_file = "resources/ijbb_metadata.csv"
metadata = np.genfromtxt(metadata_file, dtype=str, delimiter=",")

subj_bag_indices = parse_metadata(metadata)

indices = filter_by_counts(subj_bag_indices, 2)

metadata = metadata[indices]
features = features[indices]

paths = metadata[:, 0]
boxes = metadata[:, 1:5].astype(int)
subjects = metadata[:, 6].astype(int)
labels = metadata[:, 7].astype(int)

y_true = []

methods = ["method_*", "method_1", "method_2", "method_3", "method_4", "method_5"]
evaluator = Evaluator(methods)

doc = dominate.document(title='Faces by distance to median')
doc_content = div(cls='subjects')
with doc.head:
    script(src='https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js')


counter = 0
to_download = []
bags_number_hist = []
for s, subject in enumerate(np.unique(subjects)):

    idx = np.flatnonzero(subjects == subject)
    
    m = len(idx)
    if m >= 30 or counter >= 500:
        continue
    
    counter += 1
    print(counter)

    paths_subset = paths[idx]
    boxes_subset = boxes[idx]
    features_subset = features[idx]
    labels_subset = labels[idx]

    y_true.extend(labels_subset)

    unique_paths, bag_to_index, index_to_bag = parse_paths(paths_subset)
    
    bags_number = len(unique_paths)
    bags_number_hist.append(bags_number)

    to_download.extend(unique_paths)

    '''
    for i, (path, box, label) in enumerate(zip(paths_subset, boxes_subset, labels_subset)):
        image_name = get_image_name(path, i)
        
        image = Image.open(f'images/{path}')
        image = crop(image, box, 0.5)
        image = transform(image)
        
        draw = ImageDraw.Draw(image)
        color = 'green' if label == 1 else 'red'
        draw.rectangle(((0, 0), (224, 224)), outline=color, width=5)

        image.save(f'images/single/{image_name}')
    # '''

    '''
    subj_content = div(cls='subject')
    subj_content.add(div(f'Subject: {subject} | bags: {bags_number}'))

    # Method *: Median

    mu = np.median(features_subset[labels_subset.astype(bool)], axis=0, keepdims=True).T
    distances, predictions, objective = evaluator.update('method_*', features_subset, mu, bag_to_index)

    method_div = render_faces(distances, paths_subset, boxes_subset)
    subj_content.add(f'Ground true | obj: {objective:.4f}')
    subj_content.add(method_div)

    # Method 1: Median

    mu = np.median(features_subset, axis=0, keepdims=True).T
    distances, predictions, objective = evaluator.update('method_1', features_subset, mu, bag_to_index)

    method_div = render_faces(distances, paths_subset, boxes_subset)
    subj_content.add(f'Median | obj: {objective:.4f}')
    subj_content.add(method_div)

    # Method 2: Two Pass Median

    mu = np.median(features_subset[predictions.astype(bool)], axis=0, keepdims=True).T
    distances, predictions, objective = evaluator.update('method_2', features_subset, mu, bag_to_index)

    method_div = render_faces(distances, paths_subset, boxes_subset)
    subj_content.add(f'Two-Stage Median | obj: {objective:.4f}')
    subj_content.add(method_div)

    # Method 4: MILP Median

    n_components = min(len(features_subset), 2)
    pca = PCA(n_components)
    features_reduced = pca.fit_transform(features_subset)
    K = list(index_to_bag.values())
    mu, alphas = milp(features_reduced, K, False, return_alphas=True)
    mu = np.atleast_2d(mu).T
    mu = np.median(features_reduced[predictions.astype(bool)], axis=0, keepdims=True).T
    distances, predictions, objective = evaluator.update('method_4', features_reduced, mu, bag_to_index)

    method_div = render_faces(distances, paths_subset, boxes_subset)
    subj_content.add(f'MILP | obj: {objective:.4f}')
    subj_content.add(method_div)

    subj_content.add(hr())

    subj_content.attributes['milp_errors'] = np.sum(predictions != labels_subset)

    doc_content.add(subj_content)
    # '''


plt.hist(bags_number_hist, int(len(bags_number_hist) / 3))
plt.ylabel('bags per subject')
plt.tight_layout()
plt.savefig(f'{experiment_path}/bags_density.png', dpi=300)

# '''
np.savetxt(f'{experiment_path}/to_download.csv', to_download, fmt='%s', delimiter='\n')
doc.add(doc_content)

with open(f'{experiment_path}/dist_to_median.html', 'w') as f:
    f.write(doc.render())
# '''


# JS

'''
sorted = $('.subjects .subject').get().sort((a, b) => {
    return $(b)[0].getAttribute('milp_errors') - $(a)[0].getAttribute('milp_errors');
})

sorted.forEach(x => $('.subjects').append(x))
'''