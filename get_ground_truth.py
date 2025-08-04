from query_rag_ollama import query_rag
import json
import csv
from tqdm import tqdm
import pandas as pd

PROMPT = r'''You are an expert in log parsing and regular expressions. Given a log entry and the siems' 
default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields 
as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
the regex pattern. '''

def get_ground_truth(siem):
    if siem == "elastic":
        collection = "elastic_fields"
        data = 'elastic.csv'
    else:
        collection = "splunk_fields"
        data = 'splunk.csv'
    
    print(f"Loading data from {data}...")
    df = pd.read_csv(data)
    df['ground_truth_regex'] = None
    # Loop with progress bar
    print(f"Generating ground truth regex patterns for {siem} logs...")
    for item in tqdm(df.iterrows(), desc="Processing logs", unit="log"):
        log = item[1]['log_text']
        query = PROMPT + log
        regex, source = query_rag(collection, query, verbose=False)
        item[1]['ground_truth_regex'] = regex
        df.to_csv(data, index=False)
    
    print("Ground truth regex patterns generated successfully.")
    
# def get_ground_truth(siem):
#     if siem == "elastic":
#         collection = "elastic_fields"
#         data = 'elastic.json'
#     else:
#         collection = "splunk_fields"
#         data = 'splunk.json'
    
#     print(f"Loading data from {data}...")
#     with open(data, 'r') as file:
#         python_dict = json.load(file)
#     print(f"Successfully loaded {len(python_dict)} items from {data}")

#     # Loop with progress bar
#     logs = python_dict["Sheet1"]
#     print(f"Generating ground truth regex patterns for {siem} logs...")
#     for item in tqdm(logs, desc="Processing logs", unit="log"):
#         log = item['log_text']
#         query = PROMPT + log
#         regex, source = query_rag(collection, query, verbose=False)
#         item['ground_truth_regex'] = regex
#         with open(data, 'w') as file:
#             json.dump(python_dict, file, indent=2)

#     print("Ground truth regex patterns generated successfully.")

get_ground_truth("elastic")