import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
import os
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import pickle

from skfeature_job.skfeature.function.sparse_learning_based.MCFS import mcfs
from skfeature_job.skfeature.function.sparse_learning_based.MCFS import feature_ranking as mcfs_ranking
from skfeature_job.skfeature.function.similarity_based.lap_score import lap_score
from skfeature_job.skfeature.function.similarity_based.lap_score import feature_ranking as lap_score_ranking
# from skfeature_job.skfeature.utility.construct_W import construct_W

import logging


def construct_W(X):
    # default neighbor mode is 'knn' and default neighbor size is 5
    k = 5
    n_samples, n_features = np.shape(X)

    # compute pairwise euclidean distances
    D = pairwise_distances(X, n_jobs=-1)
    D **= 2
    # sort the distance matrix D in ascending order
    dump = np.sort(D, axis=1)
    idx = np.argsort(D, axis=1)
    # choose the k-nearest neighbors for each instance
    idx_new = idx[:, 0:k + 1]
    G = np.zeros((n_samples * (k + 1), 3))
    G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
    G[:, 1] = np.ravel(idx_new, order='F')
    G[:, 2] = 1
    # build the sparse affinity matrix W
    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
    bigger = np.transpose(W) > W
    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
    return W


root_path = '/home/peyman.ghasemi1'

# Configure the logger
logging.basicConfig(filename=os.path.join(root_path, 'skfeature_job/LOGS.log'), level=logging.DEBUG)

logging.info('Start of the program at:' + datetime.now().strftime("%H:%M:%S"))
# Read and prepare the dataset
all_codes_df = pd.read_csv(os.path.join(root_path, 'all_codes_list.csv'), na_filter=False)
feature_list = all_codes_df['code'].to_list()
one_hot_encoded = np.load(os.path.join(root_path, 'resampled_one_hot_data.npy'))
x_train, _, _, _ = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                                    test_size=0.33, random_state=666)
logging.info('Data loaded at:' + datetime.now().strftime("%H:%M:%S"))

x_train = x_train.astype(int)
num_selected_features = 100

# construct W
W = construct_W(x_train)
np.save(os.path.join(root_path, 'skfeature_job/W.npy'), W)
with open(os.path.join(root_path, 'skfeature_job/W.pkl'), 'wb') as file:
    pickle.dump(W, file)
logging.info('W constructed at:' + datetime.now().strftime("%H:%M:%S"))
print(W.shape)


# # MCFS Method
# W2 = mcfs(x_train, n_selected_features=num_selected_features, W=copy.deepcopy(W))
# mcfs_indices = mcfs_ranking(W2)[: num_selected_features]
#
# # write MCFS
# with open(os.path.join(root_path, 'skfeature_job/mcfs_indices.txt'), 'w') as f:
#     for s in mcfs_indices:
#         f.write(str(s) + '\n')
# best_features = [feature_list[i] for i in mcfs_indices]
# best_features.sort()
# with open(os.path.join(root_path, 'skfeature_job/mcfs_features.txt'), 'w') as f:
#     for s in best_features:
#         f.write(s + '\n')
# logging.info('MCFS feature selection done at:' + datetime.now().strftime("%H:%M:%S"))
# del W2

# lap score feature selection
scores = lap_score(x_train, W=W)
lap_indices = lap_score_ranking(scores)[: num_selected_features]
# write LAP
with open(os.path.join(root_path, 'skfeature_job/lap_indices.txt'), 'w') as f:
    for s in lap_indices:
        f.write(str(s) + '\n')
best_features = [feature_list[i] for i in lap_indices]
best_features.sort()
with open(os.path.join(root_path, 'skfeature_job/lap_features.txt'), 'w') as f:
    for s in best_features:
        f.write(s + '\n')
logging.info('Lap score feature selection done at:' + datetime.now().strftime("%H:%M:%S"))


