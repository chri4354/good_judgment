#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:15:56 2017

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



#%% load data
IFP = pd.read_csv('ifps.csv', encoding='latin-1')
#ID = pd.read_csv('all_individual_differences.tab',sep="\t")
data1 = pd.read_csv('survey_fcasts.yr1.tab',sep="\t")
data2 = pd.read_csv('survey_fcasts.yr2.tab',sep="\t")
data3 = pd.read_csv('survey_fcasts.yr3.tab',sep="\t")
data4 = pd.read_csv('survey_fcasts.yr4.tab',sep="\t")

data = data1.merge(data2,how='outer').merge(data3,how='outer').merge(data4,how='outer')
#%% Define functions to report correct response.
# return correct option
def correct_answer(ifp_id):
    # Given an IFP, return its correct answer
    mask = IFP['ifp_id'] == ifp_id
    corr = IFP['outcome'][mask]
    return corr.values.item()

# return number of options for a particular ifp
def no_opts(ifp_id):
    # Given an IFP, return its correct answer
    mask = IFP['ifp_id'] == ifp_id
    no = IFP['n_opts'][mask]
    return no.values.item()

# %%
# parallelise for loop to return the correct outcome vector 
# (dims: DF.shape[0] x 1)
inputs = range(data.shape[0])
def processInput(i):
    return correct_answer(data['ifp_id'][i])
 
num_cores = mp.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

# append result to existing data frame DF
newcol = pd.Series(results)
data.insert(10,'outcome',newcol)

#%% BAYES hierarchical aggregation
#%%
def bayes(x,opts):
    p1 = x.values[0,]
    p2 = x.values[1,]
    return pd.DataFrame((p1 * p2) / np.sum(p1*p2), index=opts)

def compute_brier(prob, truth):
    """
    Computes Brier scores according to 
    
    Inputs:
    prob: dims = n_options x 1
    truth: dims = n_options x 1
    
    Outputs: 
    """
    
    if np.shape(prob) != np.shape(truth):
        Warning('Input dimensions must be consistent!')
    
    return np.sum((truth - prob)**2)

def aggregate(q,ifp):
    print(chr(27) + "[2J")
    print(q * '*',(len(ifps) - q) * '-')
    # question data
    dat = data[data.ifp_id == ifp]
    
    # initialise working variable
    wv = pd.DataFrame(dat[['forecast_id','answer_option','value']])
    wv = wv.pivot(index='forecast_id',columns='answer_option',values='value')
    
    # initial error
    err = wv.apply(
                lambda x: compute_brier(
                        x,
                        wv.columns.values == dat.outcome.values[0]), axis=1)
    
    # recursively pair participants at random
    converged = False
    estimate = [wv.median(axis=0).values]
    brier = [err.mean(axis=0)]
    std = [err.std(axis=0)]
    while not(converged):
        # shuffle data
        wv = wv.sample(frac=1)
        # reindex
        wv.index = range(wv.shape[0])
        
        # drop one if count is odd
        if wv.shape[0] % 2 != 0:
            wv = wv.drop(wv.index[len(wv)-1]) 
        
        # pair forecasts
        g = wv.groupby(wv.index // 2)
        
        # aggregate pairs with Bayes
        wv = g.apply(bayes,wv.columns.values).unstack(level=-1)
        wv.columns = wv.columns.droplevel()
        
        # compute error (brier)
        err = wv.apply(
                lambda x: compute_brier(
                        x,
                        wv.columns.values == dat.outcome.values[0]), axis=1)
        # compute new best estimate
        estimate.append(wv.median(axis=0).values)
        # compute new mean error
        brier.append(err.mean(axis=0))
        # compute new error std
        std.append(err.std(axis=0))
        
        # check iteration condition
        if (wv.shape[0] // 2 == 0) or (err.std(axis=0)==0.):
            converged = True
        
    return wv.values, brier, std

ifps = data.ifp_id.unique()
#ifps = ifps[0:1000] # for debugging
results = Parallel(n_jobs=70)(delayed(aggregate)(q, ifp)
                                for q, ifp in enumerate(ifps))
#%% assign results to dataframe
X = pd.DataFrame(results, columns=['estimate','brier','std'])
#%% plots results
#X.apply(lambda x: plt.plot(x['brier'], alpha=.1),axis=0)
p = X['brier'].values
s = X['std'].values

plt.figure()
for i in range(len(ifps)):
    plt.scatter(p[i],s[i], alpha=.5)
    plt.xlabel('brier')
    plt.ylabel('std')

plt.figure()
for i in range(len(ifps)):
    plt.plot(p[i])
    plt.xlabel('iter.')
    plt.ylabel('brier mean')


plt.figure()
for i in range(len(ifps)):
    plt.plot(s[i]) 
    plt.xlabel('iter.')
    plt.ylabel('brier std')
plt.show()    
