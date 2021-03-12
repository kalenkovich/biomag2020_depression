#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os

from bids import BIDSLayout
import numpy as np
import mne
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean


bids_root = Path(os.environ['biomag2020_data-bids'])


layout = BIDSLayout(bids_root, validate=True, derivatives=True)
layout.derivatives


eigenvalue_files = layout.get(suffix='eigenvalues', extension='npy')


subjects = [f.entities['subject'] for f in eigenvalue_files]
sessions = [f.entities['session'] for f in eigenvalue_files]


df = (
    pd.DataFrame(
        columns=['subject', 'session'],
        data=zip(subjects, sessions))
    .reset_index()
    .rename(columns=dict(index='eigs_id')))
df.head()


eigs_all = np.stack([np.load(f) for f in eigenvalue_files])


k = 4
initial_centers = random_center_initializer(eigs_all, k).initialize()


wasserstein_metric = distance_metric(type_metric.USER_DEFINED, func=wasserstein_distance)
kmeans_instance = kmeans(data=eigs_all, initial_centers=initial_centers, metric=wasserstein_metric)

kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
final_centers = kmeans_instance.get_centers() 


cluster_assignment = pd.DataFrame(columns=['cluster_id', 'eigs_id'],
             data=[(cluster_id, eigs_id) 
                     for cluster_id, cluster in enumerate(clusters, 1)
                     for eigs_id in cluster
                    ]
            ).sort_values(by='eigs_id').reset_index(drop=True)
cluster_assignment.head()


# Number of spectra per cluster.

[len(c) for c in clusters]


# Number of subjects with distinct combinations of session clustering.

df2 = df.merge(cluster_assignment, on='eigs_id')
df2.head()


df2.groupby('subject').agg(dict(cluster_id=['min', 'max'])).value_counts().sort_index()


# Number of subject with sessions clustered into two clusters vs. one cluster.

df2.groupby('subject').agg(dict(cluster_id=['min', 'max'])).nunique(axis=1).value_counts()

