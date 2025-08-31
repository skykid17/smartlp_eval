import requests
import pcre2
from langchain_openai import ChatOpenAI
import pandas as pd

lab_llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
    )

def clean_msg(message):
    if message.startswith("```"):
        message = message[3:]
    if message.startswith("regex"):
        message = message[6:]
    if message.endswith("```"):
        message = message[:-3]
    return message.strip()

truth_df = pd.read_csv("./evaluation/ground_truth_regex.csv")
result_df = pd.read_csv("./evaluation/regex_direct.csv")

for idx, row in truth_df.iterrows():
    log_id = str(row['log_id'])
    log = str(row['log_text'])
    messages = [
        (
            "system",
            '''You are an expert in log parsing and regular expressions. Given a log entry, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields 
    as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
    Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
    outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
    the regex pattern.''',
        ),
        ("human", log),
    ]
    response = clean_msg(lab_llm.invoke(messages).content)
    print(f"log {log_id}: {response}")
    result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = response

result_df.to_csv("./evaluation/regex_direct.csv", index=False)