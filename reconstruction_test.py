import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Read the list of all icd codes
all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)

# # Read and prepare the dataset
one_hot_encoded = np.load('Processed_data/resampled_one_hot_data.npy')
train_data, test_data, _, _ = train_test_split(one_hot_encoded, np.zeros((len(one_hot_encoded), 1)),
                                               test_size=0.33, random_state=666)

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


# Loop over all selected features and train the model and evaluate the results
for filename in selected_features_paths:
    with open(filename, 'r') as f:
        lines = f.readlines()
    selected_features = [line.strip() for line in lines]
    selected_features = list(set(selected_features))
    selected_features = all_codes_df.loc[
        all_codes_df['code'].isin(selected_features), ['code', 'chapter', 'description']]
    # print(selected_features)

    # select the features
    x_train = train_data[:, selected_features.index]
    x_test = test_data[:, selected_features.index]
    y_train = train_data[:, :]
    y_test = test_data[:, :]
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # define model
    num_classes = len(all_codes_df)
    num_features = len(selected_features)
    num_hidden_layers = 64

    inputs = Input(shape=(num_features,))
    x = Dense(num_hidden_layers)(inputs)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(num_hidden_layers)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    model.fit(x_train, y_train, epochs=35, batch_size=32, verbose=1, validation_data=(x_test, y_test),
              callbacks=[early_stopping])

    # Evaluate results
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    feature_accuracy = []
    feature_f1_score = []
    feature_recall = []
    feature_precision = []
    for i in range(y_test.shape[1]):
        feature_accuracy.append(accuracy_score(y_test[:, i], y_pred[:, i]))
        feature_f1_score.append(f1_score(y_test[:, i], y_pred[:, i], zero_division=0))
        feature_recall.append(recall_score(y_test[:, i], y_pred[:, i], zero_division=0))
        feature_precision.append(precision_score(y_test[:, i], y_pred[:, i], zero_division=0))
    all_codes_df['accuracy'] = feature_accuracy
    all_codes_df['f1_score'] = feature_f1_score
    all_codes_df['recall'] = feature_recall
    all_codes_df['precision'] = feature_precision

    # Binary cross entropy
    bce = BinaryCrossentropy(from_logits=False)
    all_codes_df['binary_cross_entropy'] = bce(y_test, y_pred.astype(float)).numpy()

    all_codes_df.to_csv(results_folder + '/reconstruction_results_' + filename.split('/')[-1].split('.')[0] + '.csv',
                        index=False)


    print('Results saved for ' + filename.split('/')[-1].split('.')[0])
