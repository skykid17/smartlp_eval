from rag_mongo import RAG
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import copy

# Create RAG instance (uses defaults unless overridden)
rag = RAG()
rag.init()   # ensure indexes exist

with open("data/eval/input/retriever_golden_dataset.json", "r", encoding="utf-8") as f:
    golden_dataset = json.load(f)

# Support both top-level formats
if isinstance(golden_dataset, dict) and "test_cases" in golden_dataset:
    test_cases: List[dict] = golden_dataset["test_cases"]
elif isinstance(golden_dataset, list):
    test_cases = golden_dataset
else:
    raise RuntimeError("Unsupported golden_dataset format")

modes = [
    # ("hybrid", {"semantic_candidates": 50, "keyword_candidates": 30}),
    ("text", {"semantic_candidates": 0, "keyword_candidates": 30}),
    ("vector", {"semantic_candidates": 50, "keyword_candidates": 0}),
]

for mode_name, mode_params in modes:
    # Work on a copy so each mode writes its own file and we don't overwrite results
    print(f"Processing mode: {mode_name}")
    ds = copy.deepcopy(golden_dataset)
    test_cases: List[dict] = ds["test_cases"] if isinstance(ds, dict) and "test_cases" in ds else ds

    for case in tqdm(test_cases):
        question = case["input"]
        # Skip if the mode-specific actual_output is already present
        if case.get(f"actual_output_{mode_name}"):
            continue
        # Use RAG.query which executes the retriever and the chain (LLM)
        # This returns (answer, retrieval_context) where retrieval_context is
        # a list of dicts like {content, metadata, score}
        try:
            answer, retrieved_docs = rag.query(
                question,
                top_k=5,
                semantic_candidates=mode_params["semantic_candidates"],
                keyword_candidates=mode_params["keyword_candidates"],
                rrf_k=60,
            )
        except Exception as exc:
            # Log and continue so we capture partial results
            print(f"Failed to query for question: {question[:60]}...: {exc}")
            answer = None
            retrieved_docs = []
        
        case["actual_output"] = answer
        
        case["retrieval_context"] = [doc["content"] for doc in retrieved_docs]

        # Save augmented dataset for this mode
        out_file = f"data/eval/output/retriever_output_{mode_name}.json"
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(ds, f, indent=2, ensure_ascii=False)
