import pcre2
import re
from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from time import time
import json

TOP_K = 3
truth_df = pd.read_csv("ground_truth_regex.csv")
lab_llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
    )

lab_llm_finetuned = ChatOpenAI(
        base_url="http://192.168.125.31:8001/v1",
        api_key="EMPTY",
        model="/home/rdpuser3/Downloads/qwen-2.5-coder-finetuned",
        temperature=0.2
    )

def query_rag(collection: str, query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./chroma", verbose: bool = False):

    embeddings = HuggingFaceEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )
    
    ## Qwen25 coder 32b instruct is hosted on vLLM on 192.168.125.31:8000
    lab_llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.5
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=lab_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    if verbose:
        print(f"Answer: {result['result']}")
        print("\nSource documents:")
        for i, doc in enumerate(result['source_documents']):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:400]}...")
            if hasattr(doc, 'metadata'):
                print(f"Source: {doc.metadata}")

    return result["result"], result["source_documents"]

def clean_msg(message):
    if message.startswith("```"):
        message = message[3:]
    if message.startswith("regex"):
        message = message[6:]
    if message.startswith("\n"):
        message = message[1:]
    if message.endswith("```"):
        message = message[:-3]
    return message.strip()

def reduce(log: str, regex: str) -> str:
    # Ensure regex is a string
    if not isinstance(regex, str):
        print(f"Warning: Expected regex to be a string, got {type(regex)}. Converting to string.")
        regex = str(regex)
        
    compiled = None
    while regex:
        try:
            compiled = pcre2.compile(regex, pcre2.MULTILINE)
            if compiled.search(log):
                break
        except Exception:
            pass  # Skip invalid patterns silently
        regex = regex[:-1]
    return regex

def resolve_duplicate_capture_groups(regex: str) -> str:
    """
    Identifies duplicate named capture groups in a PCRE2 regex pattern 
    and appends incremental numbers to resolve duplicates.
    Supports both (?P<name> and (?<name> styles.
    """
    # Pattern to match named capture groups like (?P<name> or (?<name>
    pattern = re.compile(r"(\(\?P?<)(\w+)(>)")
    seen = {}
    offset = 0

    # Iterate over matches before attempting to compile
    for match in list(pattern.finditer(regex)):
        prefix, name, suffix = match.groups()
        start, end = match.span(2)

        # Check for duplicates
        if name in seen:
            seen[name] += 1
            new_name = f"{name}{seen[name]}"
            # Adjust the regex for the replacement
            regex = regex[:start + offset] + new_name + regex[end + offset:]
            offset += len(new_name) - len(name)
        else:
            seen[name] = 1
    print(f"Resolved duplicate capture groups: {seen}")
    return regex


def escape_quotes(regex: str) -> str:
    """
    Escapes unescaped single and double quotes in the regex string.
    """
    # Escape double quotes
    escaped = []
    for i, ch in enumerate(regex):
        if ch in ["'", '"'] and (i == 0 or regex[i - 1] != '\\'):
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)

def regex_direct():                                               
    result_df = pd.read_csv("regex_direct.csv")

    for idx, row in tqdm(truth_df.iterrows(), total=truth_df.shape[0], desc="Processing logs"):
        log_id = str(row['log_id'])
        log = str(row['log_text'])
        messages = [
            (
                "system",
                '''You are an expert in log parsing and regular expressions. Given a log entry, generate a pcre2 compatible 
regex pattern with named capture groups. Capture as many fields as possible. Do not capture multiple fields within a capture 
group. Do not use a 'catchall' capture group. Always use .*? within capture groups. Take into account field values with 
whitespaces. Replace all whitespaces outside of capture groups with the \s+ token. Escape any literal special characters 
and forward slashes within the regex. Return only the regex pattern.''',
            ),
            ("human", log),
        ]
        response = clean_msg(lab_llm.invoke(messages).content)
        print(f"log {log_id}: {response}")
        result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = response

    # result_df.to_csv("regex_direct.csv", index=False)


def regex_finetuned():                                               
    result_df = pd.read_csv("regex_finetuned.csv")

    for idx, row in tqdm(truth_df.iterrows(), total=truth_df.shape[0], desc="Processing logs"):
        log_id = int(row['log_id'])

        # Skip if regex already exists
        # mask = result_df['log_id'] == log_id
        # if mask.any():
        #     existing_regex = result_df.loc[mask, 'generated_regex'].values[0]
        #     if pd.notna(existing_regex) and existing_regex.strip() != "":
        #         result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = existing_regex
        #         print(f"Skipping log {log_id} as it already has a generated regex.")
        #         continue
        log = str(row['log_text'])
        messages = [
            (
                "system",
                '''You are an expert in log parsing and regular expressions. Given a log entry, generate a pcre2 compatible 
regex pattern with named capture groups. Capture as many fields as possible. Do not capture multiple fields within a capture 
group. Do not use a 'catchall' capture group. Always use .*? within capture groups. Take into account field values with 
whitespaces. Replace all whitespaces outside of capture groups with the \s+ token. Escape any literal special characters 
and forward slashes within the regex. Return only the regex pattern.''',
            ),
            ("human", log),
        ]
        response = clean_msg(lab_llm_finetuned.invoke(messages).content)
        print(f"log {log_id}: {response}")
        result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = response

    # result_df.to_csv("regex_finetuned.csv", index=False)

def regex_rag():
    result_df = pd.read_csv("regex_rag.csv")
    for idx, row in tqdm(truth_df.iterrows(), total=truth_df.shape[0], desc="Processing logs"):
        log_id = str(row['log_id'])
        log = str(row['log_text'])
        query_template = (
    "You are an expert in log parsing and regular expressions. "
    "Given a log entry and the SIEM's default fields, generate a PCRE2-compatible regex with meaningful named capture groups. "
    "Capture as many meaningful fields as possible; do not hardcode values. "
    "Do not use any special characters @.?*!, except underscore _, in capture group names. "
    "Timestamps should remain as a single group. "
    "Use .*? in capture groups, except use .* in the last group. "
    "Replace whitespace outside groups with \\s+ and escape literal special characters. "
    "Return only the regex.\n\nLog entry: "
)
        query = query_template + log
        response = clean_msg(lab_llm.invoke(query).content)
        print(f"log {log_id}: {response}")
        result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = response

    # result_df.to_csv("regex_rag.csv", index=False)

def regex_decomposed_rag():
    result_df = pd.read_csv("regex_decomposed_rag.csv")
    max_attempts = 10
    for idx, row in tqdm(truth_df.iterrows(), total=truth_df.shape[0], desc="Processing logs"):
        log_id = str(row['log_id'])
        log_text = str(row['log_text'])
        print(f"\n\nLOG {log_id}: {log_text}")
        """
    Iteratively attempts to generate a regex for a given log entry.
    """
        regex = ""
        remaining_log = log_text
        attempt = 0

        query_template = (
    "You are an expert in log parsing and regular expressions. "
    "Given a log entry and the SIEM's default fields, generate a PCRE2-compatible regex with meaningful named capture groups. "
    "Capture as many meaningful fields as possible; do not hardcode values. "
    "Do not use any special characters @.?*!, except underscore _, in capture group names. "
    "Timestamps should remain as a single group. "
    "Use .*? in capture groups, except use .* in the last group. "
    "Replace whitespace outside groups with \\s+ and escape literal special characters. "
    "Return only the regex.\n\nLog entry: "
)

        while remaining_log and attempt <= max_attempts:
            print(f"\n{'Generating' if attempt == 0 else 'Fixing'} regex (round {attempt})...")

            # Query LLM for regex suggestion
            query = query_template + remaining_log
            response = clean_msg(query_rag("elastic_fields", query, verbose=False)[0])
            print(f"Generated regex: {response}")
            reduced_regex = reduce(remaining_log, response)
            print(f"Reduced regex: {reduced_regex}")

            # Compile + match attempt
            try:
                compiled = pcre2.compile(reduced_regex, pcre2.MULTILINE)
                match = compiled.search(remaining_log)
            except Exception as e:
                print(f"Regex compilation failed: {e}")
                break

            if not match:
                print("No match found. Stopping.")
                break

            matched_part = match.group(0)
            print(f"Matched: {matched_part}")

            # Remove matched part from log
            remaining_log = remaining_log[match.end():].replace("\n", "")
            print(f"Remaining: {remaining_log}")

            # Append new regex piece
            gap = log_text[match.end():match.end()+1]
            if regex and gap.isspace():
                regex += r"\s*" + reduced_regex
            else:
                regex += reduced_regex
            print(f"Accumulated regex: {regex}")

            attempt += 1

        if attempt > max_attempts:
            print(f"Reached max attempts ({max_attempts}).")
        else:
            print(f"Regex fully matched after {attempt} rounds.")
        regex = escape_quotes(resolve_duplicate_capture_groups(regex))
        print(f"Log {log_id}'s regex: {regex}\n")
        result_df.loc[result_df['log_id'] == int(log_id), 'generated_regex'] = regex

    # result_df.to_csv("regex_decomposed_rag.csv", index=False)

def calculate_time(scenario):
    start_time = time()
    if scenario == "direct":
        regex_direct()
    if scenario == "finetuned":
        regex_finetuned()
    if scenario == "rag":
        regex_rag()
    if scenario == "decomposed_rag":
        regex_decomposed_rag()
    end_time = time()
    time_taken = end_time - start_time
    
    with open(f"results_{scenario}.json", "r") as f:
        data = json.load(f)
    data["time_taken"] = time_taken

    with open(f"results_{scenario}.json", "w") as f:
        json.dump(data, f, indent=4)

calculate_time("direct")