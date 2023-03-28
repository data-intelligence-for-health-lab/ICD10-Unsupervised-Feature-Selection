import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Read the list of all icd codes
all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)

# Read and prepare the dataset
one_hot_encoded = np.load('Processed_data/resampled_one_hot_data.npy')
mortality_outcome = np.load('Processed_data/mortality_labels.npy')
x_train, x_test, y_train, y_test = train_test_split(one_hot_encoded, mortality_outcome,
                                                    test_size=0.33, random_state=666)
# oversample minority class
oversam = RandomOverSampler(random_state=666)
x_train, y_train = oversam.fit_resample(x_train, y_train)
x_test, y_test = oversam.fit_resample(x_test, y_test)

selected_features_paths = ['selected_features/concrete_with_weights.txt',
                           'selected_features/concrete_without_weights.txt',
                           'selected_features/selected_features_AEFS.txt',
                           'selected_features/selected_features_PFA.txt',
                           'selected_features/lap_features.txt',
                           'selected_features/mcfs_features.txt']

# Results Folder
results_folder = 'selected_features/results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
results_df = pd.DataFrame(columns=['method', 'accuracy', 'f1_score', 'recall', 'precision'])

for filename in selected_features_paths:
    with open(filename, 'r') as f:
        lines = f.readlines()
    selected_features = [line.strip() for line in lines]
    selected_features = list(set(selected_features))
    selected_features = all_codes_df.loc[
        all_codes_df['code'].isin(selected_features), ['code', 'chapter', 'description']]
    # print(selected_features)

    # select the features
    selected_x_train = x_train[:, selected_features.index]
    selected_x_test = x_test[:, selected_features.index]

    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # define logistic regression model
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(selected_x_train, y_train)

    # Evaluate the model
    y_pred = logreg.predict(selected_x_test)

    model_accuracy = accuracy_score(y_test, y_pred)
    model_f1_score = f1_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)

    # Save the results
    test_results = pd.DataFrame({'method': [filename.split('/')[-1].replace('.txt', '')],
                                 'accuracy': [model_accuracy],
                                 'f1_score': [model_f1_score],
                                 'recall': [model_recall],
                                 'precision': [model_precision]})
    results_df = pd.concat([results_df, test_results], ignore_index=True)

results_df.to_csv(os.path.join(results_folder, 'mortality_prediction_results.csv'), index=False)
print(results_df)
