import pandas as pd
import os
from datetime import datetime
import constants as cons


def get_hierarchical_structure(code: str, all_codes_df: pd.DataFrame):
    """
    find the ancestors of a given code by looking to its parents
    :param all_codes_df:
    :param code:
    :return: a list containing the ancestors (until it reaches the chapter name)
    """
    hierarchy_list = []
    while code != '':
        hierarchy_list.append(code)
        try:
            code = all_codes_df.loc[all_codes_df['code'] == code, 'parent'].values[0]
        except:
            break

    return hierarchy_list


def one_hot_encode_icd_codes(codes: list[str], one_hot_col_list: list[str]):
    one_hot_row = pd.Series(0, index=one_hot_col_list)
    codes_to_encode = []
    for code in codes:
        try:
            hierarchy_list = ancestor_dic[code]
            codes_to_encode = codes_to_encode + hierarchy_list
        except KeyError as e:
            print("Wrong Code - KeyError:", e)
            continue

    one_hot_row.loc[codes_to_encode] = 1

    return one_hot_row


all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)
one_hot_col_list = all_codes_df['code'].values

# generate a lookup table of the ancestors of each code (for optimization)
ancestor_dic = {}
for code in all_codes_df['code'].values:
    ancestor_dic[code] = get_hierarchical_structure(code, all_codes_df)
print('Ancestors Lookup Table Created...')

# read the data, sort them, and save them in another file to read by chucks
df = pd.read_csv(os.path.join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_DAD_FILE), dtype=str)
num_of_samples = len(df.index)
df['ADMIT_DTTM'] = pd.to_datetime(df['ADMIT_DTTM'], format='%d%b%Y:%H:%M:%S')
df['FILE_NO'] = pd.to_numeric(df['FILE_NO'], downcast='integer')
df.sort_values(by=['FILE_NO', 'ADMIT_DTTM'], ascending=[True, True], inplace=True)
df = df.reset_index(drop=True)
df.to_csv('Processed_data/sorted_DAD.csv', index_label='index_col')
del df

# Start the process - Read the sorted dataset with chunks
print("Start Time =", datetime.now().strftime("%H:%M:%S"))
chunksize = 5000
df_iterator = pd.read_csv('Processed_data/sorted_DAD.csv', chunksize=chunksize, index_col="index_col", dtype=str)

for i, df in enumerate(df_iterator):
    df_icd = df[cons.DAD_DISEASE_CODE_COLS]
    one_hot_df = df_icd.apply(
        lambda row: one_hot_encode_icd_codes(row.dropna().tolist(), one_hot_col_list),
        axis=1)

    # Set writing mode to append after first chunk
    mode = 'w' if i == 0 else 'a'
    # Add header if it is the first chunk
    header = i == 0

    one_hot_df.to_csv('Processed_data/one_hot_encoded_DAD.csv', header=header, mode=mode)
    print(
        f'End Time for chuck {i} - {(i + 1) * chunksize}/{num_of_samples} ({round(100*(i + 1) * chunksize / num_of_samples, 2)}%) :',
        datetime.now().strftime("%H:%M:%S"))

print('All done...')
