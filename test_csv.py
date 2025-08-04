import pandas as pd

# Read the CSV
data = 'elastic.csv'
df = pd.read_csv(data)

# Add a new column with 1 for all rows
df['new_column'] = 1

# edit for specific row
df.loc[df['log_id'] == 5, 'new_column'] = 1

# Save back to CSV
df.to_csv(data, index=False)