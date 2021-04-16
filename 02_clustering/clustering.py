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
import ot


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
eigs_all = eigs_all.clip(0, 2)


n_bins = 100
bins = np.quantile(eigs_all.flatten(), np.linspace(0, 1, n_bins + 1))
bin_centers = (bins[:-1] + bins[1:]) / 2


# Matrix of distances between bins

M = np.asarray([[abs(bc2 - bc1) for bc1 in bin_centers] for bc2 in bin_centers])
M /= M.max()


# Convert samples to pmfs/pdfs

histogram_type = 'pmf'


if histogram_type == 'pmf':
    histograms = np.asarray([np.histogram(spectrum, bins=bins)[0] for spectrum in eigs_all])
    data = (histograms.T / histograms.sum(axis=1)).T
elif histogram_type == 'pdf':
    data = np.asarray([np.histogram(spectrum, bins=bins, density=True)[0] for spectrum in eigs_all])


# ## k-means

class KMeans(kmeans):
    def _kmeans__update_centers(self):
        """!
        @brief Calculate centers of clusters in line with contained objects.

        @return (numpy.array) Updated centers.

        """
        numpy = np
        
        dimension = self._kmeans__pointer_data.shape[1]
        centers = numpy.zeros((len(self._kmeans__clusters), dimension))

        for index in range(len(self._kmeans__clusters)):
            cluster_points = self._kmeans__pointer_data[self._kmeans__clusters[index], :]
            # centers[index] = cluster_points.mean(axis=0)
            centers[index] = ot.barycenter(cluster_points.T, M=self.M, reg=self.reg)
            
        return numpy.array(centers)


k = 2


initial_centers = random_center_initializer(data, k, random_state=3).initialize()
wasserstein_metric = distance_metric(type_metric.USER_DEFINED, 
                                     func=lambda x, y: wasserstein_distance(bin_centers, bin_centers, x, y))
kmeans_instance = KMeans(data=data, initial_centers=initial_centers, metric=wasserstein_metric)

kmeans_instance.M = M
kmeans_instance.reg = 0.05


kmeans_instance.process()
final_centers = kmeans_instance.get_centers()
clusters = kmeans_instance.get_clusters()


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

