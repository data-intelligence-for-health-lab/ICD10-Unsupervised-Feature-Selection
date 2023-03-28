import pandas as pd

all_codes_df = pd.read_csv('Processed_data/all_codes_list.csv', na_filter=False)

selected_features_paths = ['selected_features/concrete_with_weights.txt',
                           'selected_features/concrete_without_weights.txt',
                           'selected_features/selected_features_AEFS.txt',
                           'selected_features/selected_features_PFA.txt',
                           'selected_features/lap_features.txt',
                           'selected_features/mcfs_features.txt']

# selected_features_paths = ['selected_features/NEW2_selected_concrete_features_without_weighting.txt']

for filename in selected_features_paths:
    with open(filename, 'r') as f:
        lines = f.readlines()
    selected_features = [line.strip() for line in lines]
    selected_features = list(set(selected_features))
    selected_features.sort()

    a = all_codes_df.loc[all_codes_df['code'].isin(selected_features), ['code', 'chapter', 'rank', 'description']]
    a.to_csv(filename.replace('.txt', '.csv'), index=False)
