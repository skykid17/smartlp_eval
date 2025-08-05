import pandas as pd

try:
    print("Testing CSV reading...")
    df = pd.read_csv('elastic.csv')
    print(f"✓ Success! Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"First 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"✗ Error: {e}")
