import pcre2
import pandas as pd
import requests
from vllm import LLM
from langchain import ChatOpenAI

LOG_PROMPT = '''You are a log generator. 
Generate a realistic and random security, application, or system log line that includes the provided field example value. 
The log should look like it came from a real log file, with plausible timestamps, hostnames, IP addresses, process names, or request details.
Do not output explanations or formatting — only return the raw log line in plain text format.
The example value must appear naturally in the log, but everything else (timestamp, IDs, etc.) should be randomized.
Here are the field details:'''

REGEX_PROMPT = '''You are an expert in log parsing and regular expressions. Given a log entry and the siems' 
default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields 
as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
the regex pattern.'''

def clean_response(response: str) -> str:
    if response.startswith("```"):
        response = response[len("```"):].strip()
    if response.startswith("regex"):
        response = response[len("regex"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()
    # Remove unnecessary whitespace and newlines
    response = response.strip()
    response = response.replace("\n", "")
    return response

def query_llm(system_prompt, user_prompt) -> str:
    llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
    )

    messages = [
        (
            "system",
            system_prompt,
        ),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)

    return clean_response(response.content)

df = pd.read_csv("elastic_fields.csv")

#for all rows in dataframe
for index, row in df.iterrows():
    
    field_name = row['Field']
    field_type = row['Type']
    field_level = row['Level']
    field_description = row['Description']
    field_example = row['Example']
    field_id = row['Id']

    details = f"""
Field Name: {field_name}
Field Type: {field_type}
Field Level: {field_level}
Description: {field_description}
Example Value: {field_example}
"""

    # Generate a log that contains this field_example
    log = query_llm(LOG_PROMPT, details)
    print(f"Generated log for field {field_name}: {log}")
    # Generate the regex to extract this field and set capture group name as 'target'
    regex = query_llm(REGEX_PROMPT, log + f"Replace the target field '{field_name}' with the {field_id}")
    print(f"Generated regex for field {field_name}: {regex}")
    row['Log'] = log
    row['Regex'] = regex

# Save to csv
df.to_csv("elastic_fields_with_logs.csv", index=False)