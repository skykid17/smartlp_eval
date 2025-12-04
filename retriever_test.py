from mongo_rag import RAG
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

# Create RAG instance (uses defaults unless overridden)
rag = RAG()
rag.init()   # ensure indexes exist

with open("data/eval/input/golden_dataset.json", "r", encoding="utf-8") as f:
    golden_dataset = json.load(f)

# Support both top-level formats
if isinstance(golden_dataset, dict) and "test_cases" in golden_dataset:
    test_cases: List[dict] = golden_dataset["test_cases"]
elif isinstance(golden_dataset, list):
    test_cases = golden_dataset
else:
    raise RuntimeError("Unsupported golden_dataset format")

for case in tqdm(test_cases):
    #skip if input already has actual_output
    if case.get("actual_output"):
        continue
    question = case["input"]
    answer, retrieved_docs = rag.query(question, top_k=5)
    case["actual_output"] = answer
    content = []
    for docs in retrieved_docs:
        content.append(docs["content"])
    case["retrieval_context"] = content
    # Save augmented dataset
    out_path = Path("data/eval/output/golden_dataset_with_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(golden_dataset, f, indent=2, ensure_ascii=False)
