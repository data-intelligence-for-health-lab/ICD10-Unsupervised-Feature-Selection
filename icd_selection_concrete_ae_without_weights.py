import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from sklearn.metrics import accuracy_score



# get column names and weights of the one-hot-encoded df
all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)
feature_list = all_codes_df['code'].to_list()
class_ranks = all_codes_df['rank'].to_numpy()
class_weights = np.ones_like(class_ranks)

# Read and prepare the dataset
one_hot_encoded = np.load('Processed_data/resampled_one_hot_data.npy')
x_train, x_test, y_train, y_test = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                                    test_size=0.33, random_state=666)


# define decoder
num_features = len(feature_list)
num_hidden_layers = 64
num_selected_features = 100


def decoder(x):
    x = Dense(num_hidden_layers)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(num_hidden_layers)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(num_features, activation='sigmoid')(x)
    return x


# Train
selector = ConcreteAutoencoderFeatureSelector(K=num_selected_features,
                                              output_function=decoder,
                                              num_epochs=500,
                                              batch_size=64,
                                              tryout_limit=1,
                                              start_temp=20,
                                              min_temp=0.01,
                                              class_weights=class_weights,
                                              initial_weights=None)

model = selector.fit(x_train, x_train, x_test, x_test)

# Select Features
best_feature_idx = selector.get_support(indices=True)
best_feature_idx = np.unique(best_feature_idx)  # remove duplicates
best_features = [feature_list[i] for i in best_feature_idx]
best_features.sort()
with open('selected_features/concrete_without_weights.txt', 'w') as f:
    for s in best_features:
        f.write(s + '\n')

# Evaluate results
prediction = model.predict(x_test)
thresh = 0.5

binary_pred = prediction.copy()
binary_pred[binary_pred < thresh] = 0
binary_pred[binary_pred >= thresh] = 1

feature_accuracy = []
for i in range(x_train.shape[1]):
    feature_accuracy.append(accuracy_score(x_test[:, i], binary_pred[:, i]))
feature_accuracy_df = pd.DataFrame({'Feature': feature_list, 'Accuracy': feature_accuracy})
feature_accuracy_df.to_csv('concrete_features_accuracy_without_weighting.csv')

print(feature_accuracy_df)
