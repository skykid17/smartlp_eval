import pcre2
import pandas as pd
import re
import json


data = 'ground_truth_regex.csv'

print(f"Loading data from {data}...")
df = pd.read_csv(data)

results = {}
failed_ids = []
compiled_count = 0

total = len(df)
for idx, row in df.iterrows():
    log_id = str(row['log_id'])
    log_text = str(row['log_text'])
    regex_pattern = str(row['ground_truth_regex'])
    entry = {"compiled": False, "extracted_fields": {}}
    try:
        # Compile the regex
        pattern = re.compile(regex_pattern)
        entry["compiled"] = True
        compiled_count += 1
        # Try to extract fields
        match = pattern.search(log_text)
        if match:
            entry["extracted_fields"] = match.groupdict()
    except Exception as e:
        failed_ids.append(log_id)
    results[log_id] = entry

# Save results to JSON
with open('ground_truth_fields.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Show stats
percent = (compiled_count / total) * 100 if total else 0
print(f"Compiled successfully: {compiled_count}/{total} ({percent:.2f}%)")