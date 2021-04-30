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


# # Load data

bids_root = Path(os.environ['biomag2020_data-bids'])


layout = BIDSLayout(bids_root, validate=True, derivatives=True)
layout.derivatives


# Finds files that contain eigenvalues of the normalized Laplacian of the sensor-to-sensor correlation matrix for each session.
# The eigenvalues are calculated in the `create_eigenvalues` rule in `Snakefile`.

eigenvalue_files = layout.get(suffix='eigenvalues', extension='npy')


# We don't need to know which eigenvalues belong to which subject and session in order to run the clustering.
# This information is used later in an attempt to evaluate the results of the clustering.

subjects = [f.entities['subject'] for f in eigenvalue_files]
sessions = [f.entities['session'] for f in eigenvalue_files]


df = (
    pd.DataFrame(
        columns=['subject', 'session'],
        data=zip(subjects, sessions))
    .reset_index()
    .rename(columns=dict(index='eigs_id')))
df.head()


# ## Prepare data

# Put all the eigenvalues into a single matrix.
# 
# Since the eigenvalues are distributed in [0, 2] (property of the Laplacian), we clip them to avoid rounding problems (for example, 2,0001).

eigs_all = np.stack([np.load(f) for f in eigenvalue_files])
eigs_all = eigs_all.clip(0, 2)


# To represent the eigenspectra, we will count the number of values in certain bins.
# These bins have to be shared between all the spectra for this to work.
# Our solution is to take all the eigenvalues from all sessions together and use the quantiles as the bin limits.

n_bins = 100
bins = np.quantile(eigs_all.flatten(), np.linspace(0, 1, n_bins + 1))
bin_centers = (bins[:-1] + bins[1:]) / 2


# Matrix of distances between bins - we will need it for the function that calculate the Wasserstein barycenters.
# The `wasserstein_distance` presumably will use a similar calculation of the "earth moving" cost.

M = np.asarray([[abs(bc2 - bc1) for bc1 in bin_centers] for bc2 in bin_centers])
M /= M.max()  # we've seen a recommendation to do this for numerical reasons in a couple of places


# Convert samples to pmfs/pdfs.
# 
# Because the bins have different widths, pmf and pdf represenations of the distributions are different.
# For pmf, the number corresponding to the bin is an estimate of the probability of an eigenvalue falling in that bin.
# For pdf, that estimate is that number times the bin width.
# 
# We haven't really reached a conclusion as to which way makes more sense.
# Some things to consider:
# - The eigenvalue spectra are very dense near `1` resulting in narrow bins and thus pdfs giving larger weight to these eigenvalues.
# - We don't know which represenation `pot.barycenter` and `wasserstein_distance` expect as input. Does it matter?
# - Should we normalize the output of `pot.barycenter` so that the total probability is still `1`? Intuitively, we should.
# - A way to check the previous two points is to see how the distance changes when one of the distributions is multiplied by a number.

histogram_type = 'pmf'


if histogram_type == 'pmf':
    histograms = np.asarray([np.histogram(spectrum, bins=bins)[0] for spectrum in eigs_all])
    data = (histograms.T / histograms.sum(axis=1)).T
elif histogram_type == 'pdf':
    data = np.asarray([np.histogram(spectrum, bins=bins, density=True)[0] for spectrum in eigs_all])


# # Clustering

# ## KMeans

# The `kmeans` class from `pyclustering` allows us to use a custom distance function but not a custom cluster center calculation.
# Thus, we had to subclass it and override the corresponding method.
# We could just implement k-means from scratch but that would become increasingly more complicated to do once we switched to other clustering algorithms.

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


k = 2  # the number of clusters for the algorithm to look for


# The Wasserstein distance metric that `KMeans` will use.

wasserstein_metric = distance_metric(type_metric.USER_DEFINED, 
                                     func=lambda x, y: wasserstein_distance(bin_centers, bin_centers, x, y))

# TODO: This code is a bit fragile because we use `bin_centers` which is a public notebook-level variable.
# Couldn't find another way because the metric function will only be given the two spectra to compare.
# One other option would be to include the bin centers in the spectra themselves either simply adding them at the end
# or putting them into a second row. Not quite sure what unintended consequences this might have.


# ## Run clustering

initial_centers = random_center_initializer(data, k, random_state=3).initialize()
kmeans_instance = KMeans(data=data, initial_centers=initial_centers, metric=wasserstein_metric)

# The distance/cost matrix
kmeans_instance.M = M

# The regularization parameter
# We don't know how to set this parameter properly.
# What we do know is that larger values -> the barycenter depends less on the data.
# Too large -> the barycenters are essentially the same for any set of spectra.
kmeans_instance.reg = 0.05  

# TODO: A better option would have been to override __init__ but we couldn't figure out quickly how to do it right.


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


# # Evaluate clusters

# We don't know which subjects are depressed/healthy and in which sessions they are under ketamine or sober.
# Therefore, we can't do straight calculation of misclustered subjects/session.
# What we have to resort to are the indirect measures:
# - Did both sessions get into the *same* depressed/healthy cluster.
# - Did two session get into *different* ketamine/sober clusters.
# - Do the sizes of depressed/healthy clusters correspond to the actual numbers of depressed/healthy participants (22/14 respectively). Complicated by the fact that we had to exclude three participant for technical reasons.

# Number of clusters.
# 
# Can be smaller than `k` if at some step no spectra were assigned to one of the clusters.
# Indicative of some problems.

len(clusters)


# Number of spectra per cluster.

[len(c) for c in clusters]


# Get cluster id for each subject/session.
# 
# TODO: I hate dataframes named df even though I was the one who named them in this notebook. EK.

df2 = df.merge(cluster_assignment, on='eigs_id')
df2.head()


# Number of subjects with distinct combinations of session clustering.
# 
# Each `c_1, c_2, n` row tells us how many subjects had one session assigned to cluster `c_1` and one to cluster `c_2`.
# If `c_1 == c_2`, both sessions were assigned to the same cluster.

df2.groupby('subject').agg(dict(cluster_id=['min', 'max'])).value_counts().sort_index()


# Number of subject with sessions clustered into two clusters vs. one cluster.

df2.groupby('subject').agg(dict(cluster_id=['min', 'max'])).nunique(axis=1).value_counts()

