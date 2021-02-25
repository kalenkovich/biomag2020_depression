#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os

from bids import BIDSLayout
import numpy as np
import mne
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import random_center_initializer
import pandas as pd


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


k = 2
initial_centers = random_center_initializer(eigs_all, k).initialize()
kmeans_instance = kmeans(eigs_all, initial_centers)

kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
final_centers = kmeans_instance.get_centers()
 


cluster_assignment = pd.DataFrame(columns=['cluster_id', 'eigs_id'],
             data=[(cluster_id, eigs_id) 
                     for cluster_id, cluster in enumerate(clusters, 1)
                     for eigs_id in cluster
                    ]
            )
cluster_assignment.head()


[len(c) for c in clusters]


df = df.merge(cluster_assignment, on='eigs_id')
df.head()


df.groupby('subject').agg(dict(cluster_id=['min', 'max'])).value_counts().sort_index()


df.groupby('subject').agg(dict(cluster_id=['min', 'max'])).nunique(axis=1).value_counts()

