import simple_icd_10 as icd
import numpy as np
import pandas as pd
import os
import re
import constants as cons


# create required folders
if 'Processed_data' not in os.listdir():
    os.mkdir('Processed_data')
if 'selected_features' not in os.listdir():
    os.mkdir('selected_features')

# read the dataset
df = pd.read_csv(os.path.join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_DAD_FILE), dtype=str)
canadian_icd_10 = pd.read_csv('ICD_10_CA_DX.csv')
disease_df = df[cons.DAD_DISEASE_CODE_COLS]
unique_codes = np.unique(disease_df.to_numpy().astype(str))
all_codes_list = pd.DataFrame(columns=['code', 'parent', 'rank', 'chapter', 'description', 'is_canadian'])


def get_canadian_codes_ancestors(icd_code: str):
    """
    This code removes the least valued digit from the ICD code until finds its parent in the main ICD codes
    It is useful to detect Canadian ICD codes. In cases that the code is related to the morphology of neoplasm, it
    handles it differently.
    :param icd_code:
    :return:
    """

    # check if the code consists of just numbers
    pattern = re.compile(r'^\d+$')
    if bool(pattern.match(icd_code)) and (len(icd_code) > 3):
        return ['NEOPLASM']

    # General Cases
    # drop last digit and check general ICD-10
    new_icd_code = icd_code[:-1]
    if icd.is_valid_item(new_icd_code):
        ancestors = icd.get_ancestors(new_icd_code)
        ancestors.insert(0, new_icd_code)
    else:
        if len(new_icd_code) != 0:
            ancestors = get_canadian_codes_ancestors(new_icd_code)
            # ancestors.insert(0, new_icd_code)
        else:
            ancestors = []

    return ancestors


# find ancestors of the unique code and add them to the code list
def add_to_codes_list(icd_code):
    global all_codes_list
    if icd.is_valid_item(icd_code):
        ancestors = icd.get_ancestors(icd_code)

        if not (icd_code in all_codes_list['code'].values):
            code_data = {
                'code': icd_code,
                'parent': None if len(ancestors) == 0 else ancestors[0],  # to make sure the list is not empty
                'rank': len(ancestors),
                'chapter': icd_code if len(ancestors) == 0 else ancestors[-1],  # to make sure the list is not empty
                'description': icd.get_description(icd_code),
                'is_canadian': False
            }
            all_codes_list = pd.concat([all_codes_list, pd.DataFrame([code_data])], ignore_index=True)
            # Now do the same thing for each ancestor
            for ancestor in ancestors:
                add_to_codes_list(ancestor)

    else:
        # check if the code is in canadian list
        if icd_code in canadian_icd_10['DX_CD'].values:
            ancestors = get_canadian_codes_ancestors(icd_code)
            if not (icd_code in all_codes_list['code'].values):
                code_data = {
                    'code': icd_code,
                    'parent': None if len(ancestors) == 0 else ancestors[0],  # to make sure the list is not empty
                    'rank': len(ancestors),
                    'chapter': icd_code if len(ancestors) == 0 else ancestors[-1],  # to make sure the list is not empty
                    'description': canadian_icd_10.loc[canadian_icd_10['DX_CD'] == icd_code, 'DX_DESC'].values[0],
                    'is_canadian': True
                }
                all_codes_list = pd.concat([all_codes_list, pd.DataFrame([code_data])], ignore_index=True)
                # Now do the same thing for each ancestor
                for ancestor in ancestors:
                    add_to_codes_list(ancestor)



for code in unique_codes:
    add_to_codes_list(code)

# print(all_codes_list)
all_codes_list.to_csv('Processed_data/all_codes_list.csv')
print('Done...')
