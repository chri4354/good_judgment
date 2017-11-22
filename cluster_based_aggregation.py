#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:13:38 2017

@author: niccolop
"""

# %% import 
import sys
sys.path.insert(0, '/data/myfunctions/')
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from metacognition import type2roc
import theano.tensor as tt
import pymc3 as pm
import pickle
from datetime import datetime


#%% load data
IFP = pd.read_csv('ifps.csv', encoding='latin-1')
#%%
#ID = pd.read_csv('all_individual_differences.tab',sep="\t")
#%%
data1 = pd.read_csv('survey_fcasts.yr1.tab',sep="\t")

#%% PREPROCESSING
#%% Define preprocessing functions
# return correct option
def correct_answer(ifp_id):
    # Given an IFP, return its correct answer
    mask = IFP['ifp_id'] == ifp_id
    corr = IFP['outcome'][mask]
    return corr.values.item()

# return closing date
def date_closed(ifp_id):
    return datetime.strptime(
            IFP.date_closed[IFP.ifp_id == ifp_id].values.item(), '%m/%d/%y')

def date_transform(i):
    return datetime.strptime(
            data1.fcast_date[i], '%Y-%m-%d')

inputs = range(data1.shape[0])
num_cores = mp.cpu_count()

# %% append ground truth
# (dims: DF.shape[0] x 1)
results = Parallel(n_jobs=num_cores)(delayed(
        correct_answer)(data1['ifp_id'][i]) for i in inputs)

# append result to existing data frame DF
newcol = pd.Series(results)
data1.insert(10,'outcome',newcol)

#%% append closing date
results = Parallel(n_jobs=num_cores)(delayed(
        date_closed)(data1.ifp_id[i]) for i in inputs)
# append column
newcol = pd.Series(results)
data1.insert(12,'date_closed',newcol)

#%% format fcast_date
results = Parallel(n_jobs=num_cores)(delayed(
        date_transform)(i) for i in inputs)
# append column
newcol = pd.Series(results)
data1['fcast_date'] = newcol.values

#%% preprocess data
    
def last_prediction(x):
    end = x.wv
    x.fcast_outcome


#%% Cluster analysis
print(__doc__)
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

data = forecasts_[['ifp_id','']] #scale(digits.data)

n_samples, n_features = data.shape
n_clusters = 10

sample_size = 300

print("n_clusters: %d, \t n_samples %d, \t n_features %d"
      % (n_clusters, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    estimator.fit(data)


bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_clusters).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()