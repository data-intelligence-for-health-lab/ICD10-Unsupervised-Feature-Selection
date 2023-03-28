import sys

sys.path.append('.')

from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scipy
import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import multiprocessing as mp

from skfeature_job.skfeature.function.sparse_learning_based.MCFS import feature_ranking as mcfs_ranking
import logging

n_clusters = 5


def process_row(i, W, W_norm):
    row_start, row_end = W.indptr[i], W.indptr[i + 1]
    if row_start == row_end:
        # no nonzero elements in this row, so skip it
        return
    row_indices = W.indices[row_start:row_end]
    row_data = W.data[row_start:row_end]
    if scipy.sparse.issparse(row_data) and not row_data.nnz:
        # skip empty rows
        return

    row_data *= W_norm[i]
    row_data *= W_norm[row_indices]

    W.data[row_start:row_end] = row_data


def compute_w_i(i, n_selected_features, X, Y):
    clf = linear_model.Lars(n_nonzero_coefs=n_selected_features)
    clf.fit(X, Y[:, i])
    return clf.coef_


###########
root_path = '/home/peyman.ghasemi1'
# Configure the logger
logging.basicConfig(filename=os.path.join(root_path, 'skfeature_job/LOGS.log'), level=logging.DEBUG)
logging.info('Start of the MCFS program at:' + datetime.now().strftime("%H:%M:%S"))
# Read and prepare the dataset
all_codes_df = pd.read_csv(os.path.join(root_path, 'all_codes_list.csv'), na_filter=False)
feature_list = all_codes_df['code'].to_list()
one_hot_encoded = np.load(os.path.join(root_path, 'resampled_one_hot_data.npy'))
x_train, _, _, _ = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                    test_size=0.33, random_state=666)

X = x_train.astype(int)
n_selected_features = 100

del one_hot_encoded
del x_train
####
with open(os.path.join(root_path, 'skfeature_job/W.pkl'), 'rb') as file:
    W = pickle.load(file)
####
logging.info('Data loaded at:' + datetime.now().strftime("%H:%M:%S"))

###########
n_samples = W.shape[0]
W = scipy.sparse.csr_matrix.maximum(W, W.T)
W_norm = np.sqrt(1 / W.sum(axis=1).A.ravel())

logging.info('Norm calculated at:' + datetime.now().strftime("%H:%M:%S"))

with mp.Pool(processes=-1) as pool:
    for i in range(W.shape[0]):
        pool.apply_async(process_row, args=(i, W, W_norm))
    pool.close()
    pool.join()

logging.info('process_row calculated at:' + datetime.now().strftime("%H:%M:%S"))

W = W.maximum(W.T)

eigen_values, eigen_vectors = scipy.linalg.eigh(W.toarray())

logging.info('eigen_values calculated at:' + datetime.now().strftime("%H:%M:%S"))

Y = eigen_vectors[:, -1 * n_clusters - 1:-1] * np.sqrt(W_norm.reshape(-1, 1))

logging.info('Scipy Operation done at:' + datetime.now().strftime("%H:%M:%S"))

with open(os.path.join(root_path, 'skfeature_job/Y_mcfs.pkl'), 'wb') as file:
    pickle.dump(Y, file)
del W
del W_norm

##############
n_sample, n_feature = X.shape
W = np.zeros((n_feature, n_clusters))
with mp.Pool(processes=-1) as pool:
    results = [pool.apply_async(compute_w_i, args=(i, n_selected_features, X, Y)) for i in range(n_clusters)]
    for i, result in enumerate(results):
        W[:, i] = result.get()

logging.info('Pool operation at:' + datetime.now().strftime("%H:%M:%S"))

mcfs_indices = mcfs_ranking(W)[: n_selected_features]

logging.info('Ranking done at:' + datetime.now().strftime("%H:%M:%S"))

# write MCFS
with open(os.path.join(root_path, 'skfeature_job/mcfs_indices.txt'), 'w') as f:
    for s in mcfs_indices:
        f.write(str(s) + '\n')
best_features = [feature_list[i] for i in mcfs_indices]
best_features.sort()
with open(os.path.join(root_path, 'skfeature_job/mcfs_features.txt'), 'w') as f:
    for s in best_features:
        f.write(s + '\n')
print('MCFS feature selection done at:' + datetime.now().strftime("%H:%M:%S"))





