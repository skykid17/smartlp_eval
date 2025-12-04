import logging
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import LocalModel

vllm_model = LocalModel(
    model="qwen25-coder-32b-awq",
    base_url="http://192.168.125.31:8000/v1",
    api_key="test",
    generation_kwargs={"response_format": "json_object"},
)

def main() -> None:
    
    logging.info("Loading existing golden dataset from data/eval/output/golden_dataset_with_results.json")
    
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file("data/eval/output/golden_dataset_with_results.json")

    test_cases = []
    for golden in dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.actual_output,
            context=golden.context,
            expected_output=golden.expected_output,
            retrieval_context=golden.retrieval_context,
        )
        test_cases.append(test_case)

    evaluate(
        test_cases=test_cases,
        metrics=[ContextualPrecisionMetric(model=vllm_model),
            ContextualRecallMetric(model=vllm_model),
            ContextualRelevancyMetric(model=vllm_model)
            ],
        async_config=AsyncConfig(run_async=True, max_concurrent=2),
        display_config=DisplayConfig(file_output_dir="data/eval/output"),
    )

if __name__ == "__main__":
    main()