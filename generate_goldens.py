import argparse
import json
import logging
import random
import sys
from pathlib import Path

from openai import APITimeoutError

from deepeval.models import LocalModel
from deepeval.models.embedding_models import OllamaEmbeddingModel
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate goldens from document collection.")
    parser.add_argument("--document-count", type=int, default=100, help="How many documents to sample.")
    parser.add_argument("--output-dir", default="data/eval/input/synthetic_data", help="Where to write datasets.")
    parser.add_argument("--paths-json", default="data/eval/input/all_document_paths.json", help="Source paths list.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (keeps sampling deterministic for debugging).")
    return parser.parse_args()

def select_document_paths(paths: list[str], max_count: int) -> list[str]:
    unique = list(dict.fromkeys(paths))
    if not unique:
        logging.error("No documents were available in the selection list.")
        sys.exit(1)
    if len(unique) > max_count:
        selected = random.sample(unique, max_count)
    else:
        selected = unique[:]
        random.shuffle(selected)
    return selected

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    with open(args.paths_json, "r", encoding="utf-8") as f:
        document_paths = json.load(f)

    selected = select_document_paths(document_paths, args.document_count)
    out_file = Path(f"data/eval/input/{len(selected)}_document_paths.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    logging.info("Wrote final JSON with %d entries to %s", len(selected), out_file)

    vllm_model = LocalModel(
        model="qwen25-coder-32b-awq",
        base_url="http://192.168.125.31:8000/v1",
        api_key="test",
        generation_kwargs={
            "response_format": {"type": "json_object"},
        },
    )

    synthesizer = Synthesizer(model=vllm_model)

    try:
        goldens = synthesizer.generate_goldens_from_docs(
            max_goldens_per_context=1,
            document_paths=selected,
            context_construction_config=ContextConstructionConfig(
                critic_model=vllm_model,
                embedder=OllamaEmbeddingModel(model="all-minilm:l6-v2"),
                max_contexts_per_document=2,
            ),
        )
    except APITimeoutError as err:  # pragma: no cover - runtime handling
        logging.error(
            "LLM request timed out while generating goldens. "%
            "Please confirm %s is reachable and try again. Error: %s",
            vllm_model.base_url,
            err,
        )
        sys.exit(1)
    except Exception as err:  # pragma: no cover
        logging.exception("Unexpected error while generating goldens: %s", err)
        sys.exit(1)

    synthesizer.save_as(
        file_type="json",
        directory=args.output_dir,
        file_name="golden_dataset",
    )

if __name__ == "__main__":
    main()

