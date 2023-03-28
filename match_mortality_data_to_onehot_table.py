import numpy as np
import pandas as pd
import os
from datetime import datetime
import constants as cons

mortality_followup_period_days = 90  # days

# read and resample the DAD dataset
main_df_path = 'Processed_data/sorted_DAD.csv'
sorted_DAD_df = pd.read_csv(main_df_path, dtype=str, usecols=['FILE_NO', 'ADMIT_DTTM'])
sorted_DAD_df['ADMIT_DTTM'] = pd.to_datetime(sorted_DAD_df['ADMIT_DTTM'])
sorted_DAD_df['FILE_NO'] = pd.to_numeric(sorted_DAD_df['FILE_NO'], downcast='integer')

df_resampled = sorted_DAD_df.groupby('FILE_NO').resample('3M', on='ADMIT_DTTM').max()
df_resampled.dropna(inplace=True)
df_resampled = df_resampled.add_suffix('_MAX').reset_index()
df_resampled = df_resampled[['FILE_NO', 'ADMIT_DTTM_MAX']]


# Read VS dataset
vs_df = pd.read_csv(os.path.join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_VS_FILE), dtype=str, usecols=['FILE_NO', 'DETHDATE'])
vs_df['FILE_NO'] = pd.to_numeric(vs_df['FILE_NO'], downcast='integer')
vs_df['DETHDATE'] = pd.to_datetime(vs_df['DETHDATE'], format='%d%b%Y:%H:%M:%S')
vs_df.drop_duplicates(subset=['FILE_NO'], inplace=True)

# Join the two dataframes on the FILE_NO column
merged_df = pd.merge(df_resampled, vs_df, on='FILE_NO', how='left')

# Fill any missing DETHDATE values with a very large datetime value to represent that the patient did not die
merged_df['DETHDATE'] = merged_df['DETHDATE'].fillna(pd.Timestamp.max)

# Create a new column 'DEATH' in merged_df based on the time difference between ADMIT_DTTM_MAX and DETHDATE
merged_df['DEATH'] = (merged_df['DETHDATE'] - merged_df['ADMIT_DTTM_MAX']) <= pd.Timedelta(days=mortality_followup_period_days)

# Drop the ADMIT_DTTM and DETHDATE columns
merged_df = merged_df.drop(['ADMIT_DTTM_MAX', 'DETHDATE'], axis=1)

np.save('Processed_data/mortality_labels.npy', merged_df['DEATH'].to_numpy())
