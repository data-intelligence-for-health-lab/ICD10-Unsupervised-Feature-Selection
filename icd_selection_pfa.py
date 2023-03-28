import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans


def pfa_selector(A, k, debug=False):
    class PFA(object):
        def __init__(self, n_features, q=None):
            self.q = q
            self.n_features = n_features

        def fit(self, X):
            if self.q is None:
                self.q = int(X.shape[1]/2)
            print(self.q)
            sc = StandardScaler()
            X = sc.fit_transform(X)

            ipca = IncrementalPCA(n_components=self.q, batch_size=2*X.shape[1]).fit(X)
            print('PCA Captured Variance: ', np.sum(ipca.explained_variance_ratio_), ' with ', self.q, ' components')

            self.n_components_ = ipca.n_components_
            A_q = ipca.components_.T

            kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
            clusters = kmeans.predict(A_q)
            cluster_centers = kmeans.cluster_centers_
            print('KMeans Captured Variance: ', kmeans.inertia_)

            self.indices_ = []
            for cluster_idx in tqdm(range(self.n_features)):
                indices_in_cluster = np.where(clusters == cluster_idx)[0]
                points_in_cluster = A_q[indices_in_cluster, :]
                centroid = cluster_centers[cluster_idx]
                distances = np.linalg.norm(points_in_cluster - centroid, axis=1)
                optimal_index = indices_in_cluster[np.argmin(distances)]
                self.indices_.append(optimal_index)

    pfa = PFA(n_features=k)
    pfa.fit(A)
    if debug:
        print('Performed PFW with q=', pfa.n_components_)
    column_indices = pfa.indices_
    return column_indices


# generate the weights
# get column names and weights of the one-hot-encoded df
all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)
feature_list = all_codes_df['code'].to_list()



# Read and prepare the dataset
one_hot_encoded = np.load('Processed_data/resampled_one_hot_data.npy')
x_train, x_test, y_train, y_test = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                                    test_size=0.33, random_state=666)

num_features = len(feature_list)
num_selected_features = 100

indices = pfa_selector(x_train, num_selected_features)

with open('selected_feature_indices_PFA.txt', 'w') as f:
    for s in indices:
        f.write(str(s) + '\n')

best_features = [feature_list[i] for i in indices]
best_features.sort()
with open('selected_features/selected_features_PFA.txt', 'w') as f:
    for s in best_features:
        f.write(s + '\n')
