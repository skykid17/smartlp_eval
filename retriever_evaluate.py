from deepeval import evaluate
import json
from tqdm import tqdm
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.evaluate import AsyncConfig

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