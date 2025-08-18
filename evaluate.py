import sys

# Ensure a modern SQLite for Chroma by shimming pysqlite3 before any imports that may load sqlite3
try:
    import pysqlite3.dbapi2 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3
except Exception:
    pass

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from chromadb import Settings

TOP_K = 3

PROMPT = r'''You are an expert in log parsing and regular expressions. Given a log entry and the siems' 
default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields 
as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
the regex pattern. '''

# ==== Placeholder for Regex Generation ====
def query_rag(collection: str, query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", verbose: bool = False):
    embeddings = HuggingFaceEmbeddings(model=embedding_model, model_kwargs={"device": "cpu"})
    vectorstore = Chroma(
        client_settings=Settings(is_persistent=True, anonymized_telemetry=False),
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )
    # Qwen25 coder 32b instruct is hosted on vLLM on localhost:8000
    lab_llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
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


# ==== Load Ground Truth Data ====
def load_ground_truth():
    ground_truth_regex = {}
    with open("ground_truth_regex.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth_regex[row["log_id"]] = row["ground_truth_regex"]

    with open("ground_truth_fields.json", encoding="utf-8") as f:
        ground_truth_fields = json.load(f)

    return ground_truth_regex, ground_truth_fields


# ==== Metrics ====
def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def exact_match_accuracy(generated_regex, ground_truth_regex):
    return 1.0 if generated_regex == ground_truth_regex else string_similarity(generated_regex, ground_truth_regex)

def functional_accuracy(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = re.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0.0
        extracted = match.groupdict()
        correct = 0
        for field, gt_value in ground_truth_fields.items():
            if field in extracted and extracted[field] == gt_value:
                correct += 1
        return correct / len(ground_truth_fields) if ground_truth_fields else 0.0
    except re.error:
        return 0.0

def field_level_precision_recall(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = re.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0.0, 0.0
        extracted = match.groupdict()
        correct_fields = set(extracted.keys()) & set(ground_truth_fields.keys())
        true_positive = 0
        for field in correct_fields:
            sim = string_similarity(str(extracted[field]), str(ground_truth_fields[field]))
            if sim > 0.9:  # high similarity threshold
                true_positive += 1
        precision = true_positive / len(extracted) if extracted else 0.0
        recall = true_positive / len(ground_truth_fields) if ground_truth_fields else 0.0
        return precision, recall
    except re.error:
        return 0.0, 0.0

def compilation_success(generated_regex):
    try:
        re.compile(generated_regex)
        return 1.0
    except re.error:
        return 0.0

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

# ==== Evaluation ====
def evaluate_model(scenario_name, ground_truth_regex, ground_truth_fields):
    results = []
    output_file = f"{scenario_name}_model.csv"
    df = pd.read_csv(output_file)
    if 'generated_regex' not in df.columns:
        df['generated_regex'] = None
    
    df['generated_regex'] = df['generated_regex'].astype("string")
    for idx, row in tqdm(df.iterrows(), desc="Processing logs", unit="log", total=len(df)):
        if pd.notna(row['generated_regex']):
            print(f"Skipping index {idx} as generated_regex is already set.")
            continue
        log_id = str(row['log_id'])
        log_text = str(row['log_text'])
        regex, sources = query_rag("elastic_fields", PROMPT + log_text)
        cleaned_regex = clean_regex(regex)
        df.at[idx, 'generated_regex'] = cleaned_regex
        print(f"[UPDATE] log_id={log_id} | generated_regex={cleaned_regex}")  # Real-time update
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)  # Write after each update

        gt_fields = ground_truth_fields[log_id]["extracted_fields"]

        em_acc = exact_match_accuracy(cleaned_regex, gt_regex)
        func_acc = functional_accuracy(cleaned_regex, log_text, gt_fields)
        precision, recall = field_level_precision_recall(cleaned_regex, log_text, gt_fields)
        comp_rate = compilation_success(cleaned_regex)

        results[log_id] = {
            "Exact_Match_Accuracy": em_acc,
            "Functional_Accuracy": func_acc,
            "Field_Precision": precision,
            "Field_Recall": recall,
            "Compilation_Success": comp_rate
        }
        # Save results to json
        with open(f"{scenario_name}_results.json", "w") as json_file:
            json.dump(results, json_file)
        
    return results

# ==== Main Execution ====
if __name__ == "__main__":
    gt_regex, gt_fields = load_ground_truth()

    metrics = evaluate_model("direct_prompting", gt_regex, gt_fields)
    print(f"\nResults for direct_prompting:")
    for m in metrics[:3]:  # Print first 3 for preview
        print(m)