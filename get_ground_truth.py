import sys

# Ensure a modern SQLite for Chroma by shimming pysqlite3 before any imports that may load sqlite3
try:
    import pysqlite3.dbapi2 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3
except Exception:
    pass

import json
import csv
from tqdm import tqdm
import pandas as pd
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from chromadb import Settings

load_dotenv()

PROMPT = r'''You are an expert in log parsing and regular expressions. Given a log entry and the siems' 
default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields 
as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
the regex pattern. '''

TOP_K = 3

def query_rag(collection: str, query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./chroma", verbose: bool = False):

    embeddings = HuggingFaceEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        client_settings=Settings(is_persistent=True, anonymized_telemetry=False),
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )
    
    ollama_llm = ChatOllama(
        model="qwen2.5-coder",
        temperature=0.7
    )

    local_llm = ChatOpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN"),
        model="openai/gpt-4.1",
        temperature=0.7
    )

    ## Qwen25 coder 32b instruct is hosted on vLLM on 192.168.125.32:8000
    lab_llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_llm,
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

def clean_regex(regex):
    # If starts with ```regex, remove it
    if regex.startswith("```regex"):
        regex = regex[len("```regex"):].strip()
    # if ends with ```, remove it
    if regex.endswith("```"):
        regex = regex[:-len("```")].strip()
    # Remove unnecessary whitespace and newlines
    regex = regex.strip()
    regex = regex.replace("\n", "")
    return regex

def get_ground_truth(siem):
    if siem == "elastic":
        collection = "elastic_fields"
        data = 'elastic.csv'
    else:
        collection = "splunk_fields"
        data = 'splunk.csv'
    
    print(f"Loading data from {data}...")
    df = pd.read_csv(data)
    if 'ground_truth_regex' not in df.columns:
        df['ground_truth_regex'] = None
    
    # Loop with progress bar
    print(f"Generating ground truth regex patterns for {siem} logs...")
    for index, row in tqdm(df.iterrows(), desc="Processing logs", unit="log", total=len(df)):
        # If ground_truth_regex is already set, skip
        if pd.notna(row['ground_truth_regex']):
            print(f"Skipping index {index} as ground_truth_regex is already set.")
            continue
        log = row['log_text']
        query = PROMPT + log
        regex, source = query_rag(collection, query)
        df.at[index, 'ground_truth_regex'] = clean_regex(regex)
        df.to_csv(data, index=False, quoting=csv.QUOTE_ALL)
    
    # Save the updated dataframe
    print("Ground truth regex patterns generated successfully.")


get_ground_truth("elastic")