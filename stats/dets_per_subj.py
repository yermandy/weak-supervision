import numpy as np
import matplotlib.pyplot as plt

metadata = np.genfromtxt('resources/ijbb_metadata.csv', dtype=str, delimiter=',')

subjects = metadata[:, 6].astype(int)

distribution = []

subjects_unique, subjects_indices, subjects_counts = np.unique(subjects, return_index=True, return_counts=True)

subjects_counts = subjects_counts[subjects_counts < 500]

bins = len(subjects_counts)

detections_per_subject_mean = np.mean(subjects_counts)
detections_per_subject_median = np.median(subjects_counts)

plt.vlines(detections_per_subject_mean, 0, 90, color='r', linewidth=2, label=f'mean: {detections_per_subject_mean:.0f}')
plt.vlines(detections_per_subject_median, 0, 90, color='magenta', linewidth=2, label=f'median: {detections_per_subject_median:.0f}')
plt.tight_layout()
plt.legend()
plt.hist(subjects_counts, int(bins / 10)) 
plt.savefig('results/detections_per_subject.png', dpi=300)

