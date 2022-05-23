from __future__ import print_function
from lightfm import LightFM
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from analysis_functions import create_sparse_matrix
from analysis_functions import calculate_sparsity
from analysis_functions import split_train_test_per_user
from analysis_functions import Baseline
from analysis_functions import pct_masked
from analysis_functions import evaluate
import os
import shutil
import sys
import logging
import scrapbook as sb
from recommenders.utils.python_utils import binarize
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from recommenders.models.sar import SAR


import numpy as np
from scipy import sparse

import seaborn as sn
sn.set()

import tensorflow
from tensorflow.python.keras.regularizers import L2


import bottleneck as bn


TOP_K = 20
# Import main dataset
# for colab
df = pd.read_csv('/content/drive/MyDrive/music/lastfm_9000_users.csv', na_filter=False)
# df = pd.read_csv('lastfm_9000_users.csv', na_filter=False)
df = df.drop(['Unnamed: 0'], axis=1)

# Proportion of invalid values
len(df.loc[df.artist_mbid == ''])/len(df)

df = df[df.artist_mbid != '']
df = df[df.artist_name != '']

# For init purpose
#df = df[df['plays'] > 20]
# For colab runtime
df = df[df['plays'] > 150]

df.hist(column='plays')

sns.distplot(df.plays)

sns.distplot(df[df.plays < 1000].plays)
plt.title("Distribution of Plays (subset of under 1000 plays)")
plt.xlabel("Number of Plays")
plt.ylabel("Density")


"""
<user_id,artist_mbid,artist_name,plays>
"""

# create sparse matrix
plays_sparse = create_sparse_matrix(df).astype('float')
print('Matrix Sparsity:', calculate_sparsity(plays_sparse))

# Split data into training and test sets
train_base, test_base, user_count = split_train_test_per_user(plays_sparse, k=3, interactions=10)
print("Percentage of original data masked:", pct_masked(plays_sparse, train_base.T.tocsr()))
print("Users masked:", user_count)

model_baseline = Baseline(n_recs=20)

model_baseline.fit(train_base)

coverage, precision, recall, ndcg = evaluate(model_baseline, "baseline", test_base, plays_sparse)
print("Precision:", precision)
print("Recall:", recall)
print("Coverage:", coverage)
print("Average NDCG per User:", ndcg)

final_results = {'model': ['baseline'], 'precision (%)': [precision], 'recall (%)': [recall],
                 'coverage (%)': [coverage], 'ndcg (%)': [ndcg]}

# model_als = implicit.als.AlternatingLeastSquares(factors=30, regularization=0.01)
#
# # Train model
# print("Fitting model...")
# model_als.fit(train)


# coverage, precision, recall, ndcg = evaluate(model_als, "implicit", test, train.T.tocsr())
# print("Precision:", precision*100, '%')
# print("Recall:", recall*100, '%')
# print("Coverage:", coverage*100, '%')
# print("Average NDCG per User:", ndcg*100, '%')
#
# final_results['model'].append('als')
# final_results['precision (%)'].append(precision*100)
# final_results['recall (%)'].append(recall*100)
# final_results['coverage (%)'].append(coverage*100)
# final_results['ndcg (%)'].append(ndcg*100)


######################################################################################################################
# # SAR
#
# Data preprocessing
print('##############################################################################################################')
plays = df.plays
plays_log = np.log(plays+1)
plays_norm = (plays_log - plays_log.mean())/plays_log.std()
df['plays_norm'] = plays_norm
df.drop(['plays'], axis=1, inplace=True)

# convert precision to 32-bit to save memory
timestamp = [i for i in range(len(df.user_id))]
df['timestamp'] = timestamp
df['timestamp'] = df['timestamp'].astype(np.uint8)


train_sar, test_sar = python_stratified_split(df, ratio=0.6, col_user='user_id', col_item='artist_mbid', seed=0)
print("""
Train:
Total Ratings: {train_total}
Unique Users: {train_users}
Unique Items: {train_items}

Test:
Total Ratings: {test_total}
Unique Users: {test_users}
Unique Items: {test_items}
""".format(
    train_total=len(train_sar),
    train_users=len(train_sar['user_id'].unique()),
    train_items=len(train_sar['artist_mbid'].unique()),
    test_total=len(test_sar),
    test_users=len(test_sar['user_id'].unique()),
    test_items=len(test_sar['artist_mbid'].unique()),
))
# load sar model


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SAR(
    col_user="user_id",
    col_item="artist_mbid",
    col_rating="plays_norm",
    col_timestamp="timestamp",
    similarity_type="jaccard",
    time_decay_coefficient=30,
    timedecay_formula=True,
    normalize=True
)

# train


with Timer() as train_time:
    model.fit(train_sar)

print("Took {} seconds for training.".format(train_time.interval))

with Timer() as test_time:
    top_k = model.recommend_k_items(test_sar, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))


eval_map = map_at_k(test_sar, top_k, col_user='user_id', col_item='artist_mbid', col_rating='plays_norm', k=TOP_K)
eval_ndcg = ndcg_at_k(test_sar, top_k, col_user='user_id', col_item='artist_mbid', col_rating='plays_norm', k=TOP_K)
eval_precision = precision_at_k(test_sar, top_k, col_user='user_id', col_item='artist_mbid', col_rating='plays_norm', k=TOP_K)
eval_recall = recall_at_k(test_sar, top_k, col_user='user_id', col_item='artist_mbid', col_rating='plays_norm', k=TOP_K)

test_sar.replace([np.inf, -np.inf], np.nan, inplace=True)
test_sar.dropna(inplace=True)

print("Model:\t",
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall,
      sep='\n')

######################################################################################################################





