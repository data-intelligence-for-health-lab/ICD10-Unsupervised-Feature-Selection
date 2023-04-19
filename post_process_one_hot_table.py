import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os


one_hot_df_path = 'Processed_data/one_hot_encoded_DAD.csv'
main_df_path = 'Processed_data/sorted_DAD.csv'

# get column names of the one-hot-encoded df
one_hot_sample = pd.read_csv(one_hot_df_path, index_col=0, nrows=1)
one_hot_cols = one_hot_sample.columns

# Initialize an empty list to store the arrays
arrays = []


def func(x, itr_num):
    """
    This function takes a chunk of the main one-hot encoded dataframe and processes it (by resampling to 3 months)
    :param x: chunk of the main one-hot encoded dataframe
    :param itr_num: which iteration is this (used for debugging)
    :return: 0 -- it appends the processed array to the global list of arrays
    """
    global arrays
    skiprows = x.index[0] + 1
    nrows = x.index[-1] - x.index[0] + 1

    one_hot_df_group = pd.read_csv(one_hot_df_path, index_col=0, skiprows=skiprows, nrows=nrows, names=one_hot_cols)
    df2 = pd.concat([x, one_hot_df_group], axis=1)
    df_resampled = df2.groupby('FILE_NO').resample('3M', on='ADMIT_DTTM').max()
    df_resampled.dropna(inplace=True)

    # # Write to CSV
    # # Set writing mode to append after first chunk
    # mode = 'w' if itr_num == 0 else 'a'
    # # Add header if it is the first chunk
    # header = itr_num == 0
    # df_resampled.to_csv('one_hot_encoded_resampled.csv', header=header, mode=mode)

    # Store in a Numpy array
    bool_chunk = df_resampled[one_hot_cols].astype(bool)  # Convert the dataframe to boolean values
    arrays.append(bool_chunk.values)  # Append the boolean array to the list of arrays

    return 0


# Read the main dataframe
df = pd.read_csv(main_df_path, dtype=str, usecols=['FILE_NO', 'ADMIT_DTTM'])
df['ADMIT_DTTM'] = pd.to_datetime(df['ADMIT_DTTM'])
df['FILE_NO'] = pd.to_numeric(df['FILE_NO'], downcast='integer')

# find unique patients to iterate over them (to avoid a situation that one patient is split into multiple chunks)
unique_patients = df['FILE_NO'].unique()

# iterate over certain number of unique patients and analyze their data
print("Start Time =", datetime.now().strftime("%H:%M:%S"))

batch_size = 2000  # how many patients in each batch should be processed (we have about 5 samples per patients)
for i in tqdm(range(0, len(unique_patients), batch_size)):
    i2 = min(i + batch_size, len(unique_patients))  # to handle the last batch
    batch = unique_patients[i:i2]
    batch_df = df.loc[df['FILE_NO'].isin(batch)]
    func(batch_df, i)

# Concatenate the arrays along the rows axis
np_result = np.concatenate(arrays, axis=0)
print(np_result.shape)
np.save('Processed_data/resampled_one_hot_data', np_result)

print("End Time =", datetime.now().strftime("%H:%M:%S"))
print('Done')
