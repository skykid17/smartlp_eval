import os
from tqdm import tqdm
import pandas as pd

df = pd.read_csv('elastic.csv')
# df['ground_truth_regex'] = None
# df_modified = df.drop(columns=['new_column'])

# #save the modified dataframe
# df_modified.to_csv('elastic.csv', index=False)

for index, row in tqdm(df.iterrows(), desc="Processing logs", unit="log", total=len(df)):
    # If ground_truth_regex is already set, skip
    if pd.notna(row['ground_truth_regex']):
        print(f"Skipping index {index} as ground_truth_regex is already set.")
        continue
    print("Processing index:", index)

