from deepeval import evaluate
import json
from tqdm import tqdm
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.evaluate import AsyncConfig
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# load output file
BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_DIR   = BASE_DIR / "data" / "eval" / "output"
OUTPUT_DIR  = BASE_DIR / "data" / "eval" / "output"

modes = ["hybrid", "text", "vector"]

def build_test_case(item):
    return LLMTestCase(
        input=item.get("input", ""),
        actual_output=item.get("actual_output", ""),
        expected_output=item.get("expected_output", ""),
        context=item.get("context", ""),
        retrieval_context=item.get("retrieval_context", []),
    )

precision_metric = ContextualPrecisionMetric(async_mode=False)
recall_metric = ContextualRecallMetric(async_mode=False)
relevancy_metric = ContextualRelevancyMetric(async_mode=False)

for mode in modes:
    test_cases = []
    with open(f"data/eval/output/output_{mode}.json", "r", encoding="utf-8") as f:
        retriever_outputs = json.load(f)
    for item in tqdm(retriever_outputs, desc=f"Building test cases for mode: {mode}"):
        test_cases.append(build_test_case(item))
    evaluate(
        test_cases=test_cases, metrics=[precision_metric, recall_metric, relevancy_metric], async_config=AsyncConfig(run_async=False)
    )

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def find_conciseness(input_text, output_text):
    input_embedding = embedder.encode(input_text)
    output_embedding = embedder.encode(output_text)
    return round(cosine(input_embedding, output_embedding), 2) #2 decimal places

def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom > 0 else 0.0


def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

for mode in modes:
    output_data = load_results(OUTPUT_DIR / f"retriever_output_{mode}.json")
    results_data = load_results(OUTPUT_DIR / f"retriever_results_{mode}.json")
    for i in tqdm(range(len(output_data)), desc="Calculating conciseness"):
        actual_output = output_data[i]["actual_output"]
        expected_output = output_data[i]["expected_output"]
        conciseness_score = find_conciseness(actual_output, expected_output)
        results_data[i]["conciseness"] = conciseness_score
    with open(OUTPUT_DIR / f"retriever_results_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)

print("Done")