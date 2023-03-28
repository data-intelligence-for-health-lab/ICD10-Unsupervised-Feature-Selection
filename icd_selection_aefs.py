import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import tensorflow.compat.v1 as tf
from tqdm import tqdm

tf.disable_v2_behavior()


def next_batch(samples, labels, num):
    # Return a total of `num` random samples and labels.
    idx = np.random.choice(len(samples), num)

    return samples[idx], labels[idx]


def standard_single_hidden_layer_autoencoder(X, units, O):
    reg_alpha = 1e-3
    D = X.shape[1]
    weights = tf.get_variable("weights", [D, units])
    biases = tf.get_variable("biases", [units])
    X = tf.nn.leaky_relu(tf.matmul(X, weights) + biases)
    X = tf.layers.dense(X, O, tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(reg_alpha))
    return X, weights


def aefs_subset_selector(train, K, epoch_num=1000, alpha=0.1):
    D = train.shape[1]
    O = train.shape[1]
    learning_rate = 0.001

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, (None, D))
    TY = tf.placeholder(tf.float32, (None, O))
    Y, weights = standard_single_hidden_layer_autoencoder(X, K, O)

    loss = tf.reduce_mean(tf.square(TY - Y)) + alpha * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(weights), axis=1)),
                                                                     axis=0) + tf.losses.get_total_loss()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    batch_size = 256
    batch_per_epoch = train[0].shape[0] // batch_size

    costs = []

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as sess:
        sess.run(init)
        for ep in tqdm(range(epoch_num)):
            cost = 0
            for batch_n in range(batch_per_epoch):
                imgs, yimgs = next_batch(train, train, batch_size)
                _, c, p = sess.run([train_op, loss, weights], feed_dict={X: imgs, TY: yimgs})
                cost += c / batch_per_epoch
            costs.append(cost)

    return list(np.argmax(np.abs(p), axis=0)), costs


# generate the weights
# get column names and weights of the one-hot-encoded df
all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)
feature_list = all_codes_df['code'].to_list()
class_ranks = all_codes_df['rank'].to_numpy()
class_weights = 1 / (class_ranks + 1)


# Read and prepare the dataset
one_hot_encoded = np.load('Processed_data/resampled_one_hot_data.npy')
x_train, x_test, y_train, y_test = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                                    test_size=0.33, random_state=666)

num_features = len(feature_list)
num_selected_features = 100

indices, _ = aefs_subset_selector(x_train, num_selected_features)

with open('selected_feature_indices_AEFS.txt', 'w') as f:
    for s in indices:
        f.write(str(s) + '\n')

best_features = [feature_list[i] for i in indices]
best_features.sort()
with open('selected_features/selected_features_AEFS.txt', 'w') as f:
    for s in best_features:
        f.write(s + '\n')
