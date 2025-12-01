import argparse
import logging
import math
from pathlib import Path
from typing import List, Dict, Any

from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase
from deepeval.models import LocalModel

from pymongo import MongoClient
from pymongo.collection import Collection

from rag_mongo import (
    MongoHybridRetriever,
    generate_embeddings,
    DEFAULT_TEXT_PATHS,
)


def connect_mongo(uri: str, database: str, collection: str) -> Collection:
    client = MongoClient(uri)
    return client[database][collection]


def sample_documents(collection: Collection, limit: int = 1000) -> List[Dict[str, Any]]:
    # Prefer documents that actually have content
    cursor = collection.aggregate([
        {"$match": {"content": {"$exists": True, "$type": "string"}}},
        {"$sample": {"size": limit}},
    ])
    return list(cursor)


def build_corpus_for_synthesizer(docs: List[Dict[str, Any]], max_docs: int = 100) -> List[str]:
    corpus: List[str] = []
    for doc in docs[:max_docs]:
        content = doc.get("content", "")
        if content:
            corpus.append(content)
    return corpus


def generate_golden_dataset(
    collection: Collection,
    output_path: Path,
    vllm_model: str,
    vllm_base_url: str,
    num_docs_sample: int = 1000,
    max_docs_for_corpus: int = 100,
    num_questions: int = 50,
) -> List[LLMTestCase]:
    logging.info("Sampling %d documents from MongoDB for golden dataset", num_docs_sample)
    docs = sample_documents(collection, limit=num_docs_sample)
    corpus = build_corpus_for_synthesizer(docs, max_docs=max_docs_for_corpus)
    if not corpus:
        raise RuntimeError("No suitable documents found to build synthesizer corpus")

    logging.info("Building synthesizer with vllm model '%s'", vllm_model)
    synthesizer = Synthesizer(
        model=LocalModel(model=vllm_model, base_url=vllm_base_url, api_key="test")
    )

    contexts = [[text] for text in corpus]
    max_goldens_per_context = max(1, math.ceil(num_questions / max(1, len(contexts))))
    logging.info(
        "Generating up to %d synthetic test cases from %d contexts",
        num_questions,
        len(contexts),
    )
    generated_goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=max_goldens_per_context,
    )

    # Flatten the generated goldens into DeepEval LLMTestCase objects
    test_cases: List[LLMTestCase] = []
    for golden in generated_goldens:
        if not golden.input:
            continue
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                reference_contexts=golden.context,
            )
        )
        if len(test_cases) >= num_questions:
            break

    if not test_cases:
        raise RuntimeError("Synthesizer did not return any test cases")

    golden_dataset = Golden(test_cases=test_cases)
    golden_dataset.to_json(str(output_path))
    logging.info(
        "Golden dataset written to %s with %d test cases",
        output_path,
        len(test_cases),
    )
    return test_cases


def text_search_retrieve(collection: Collection, query: str, top_k: int, text_index: str) -> List[str]:
    pipeline = [
        {
            "$search": {
                "index": text_index,
                "text": {"query": query, "path": DEFAULT_TEXT_PATHS},
            }
        },
        {"$limit": top_k},
        {"$project": {"content": 1}},
    ]
    results = list(collection.aggregate(pipeline))
    return [doc.get("content", "") for doc in results]


def vector_search_retrieve(
    collection: Collection,
    query: str,
    top_k: int,
    vector_index: str,
    embedding_dim: int,
) -> List[str]:
    query_vector = generate_embeddings([query], embedding_dim)[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k,
                "limit": top_k,
            }
        },
        {"$project": {"content": 1}},
    ]
    results = list(collection.aggregate(pipeline))
    return [doc.get("content", "") for doc in results]


def hybrid_rrf_retrieve(
    retriever: MongoHybridRetriever,
    query: str,
) -> List[str]:
    docs = retriever.get_relevant_documents(query)
    return [d.page_content for d in docs]


def evaluate_retrievers(
    collection: Collection,
    vector_index: str,
    text_index: str,
    embedding_dim: int,
    golden_cases: List[LLMTestCase],
    top_k: int = 5,
):
    # Build a hybrid retriever instance mirroring rag_mongo query defaults
    retriever = MongoHybridRetriever(
        collection=collection,
        embedding_dim=embedding_dim,
        vector_index=vector_index,
        text_index=text_index,
        top_k=top_k,
        semantic_candidates=50,
        keyword_candidates=30,
        rrf_k=60,
        allowed_text_paths=DEFAULT_TEXT_PATHS,
        filter_category=None,
    )

    # Build metrics once; they will be applied across all variants
    metrics = [
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
        ContextualRelevancyMetric(),
    ]

    # Build three sets of DeepEval test cases, one per retrieval strategy
    def build_test_cases(strategy: str) -> List[LLMTestCase]:
        cases: List[LLMTestCase] = []
        for base_tc in golden_cases:
            query = base_tc.input
            if not query:
                continue

            if strategy == "text":
                retrieved = text_search_retrieve(collection, query, top_k, text_index)
            elif strategy == "vector":
                retrieved = vector_search_retrieve(collection, query, top_k, vector_index, embedding_dim)
            elif strategy == "hybrid":
                retrieved = hybrid_rrf_retrieve(retriever, query)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # DeepEval RAG retrieval uses `retrieved_contexts` on the test case
            tc = LLMTestCase(
                input=query,
                retrieved_contexts=retrieved,
                expected_output=base_tc.expected_output,
                reference_contexts=base_tc.reference_contexts,
            )
            cases.append(tc)
        return cases

    strategies = {
        "text_search": build_test_cases("text"),
        "vector_search": build_test_cases("vector"),
        "hybrid_rrf": build_test_cases("hybrid"),
    }

    for name, cases in strategies.items():
        logging.info("Evaluating strategy: %s (%d cases)", name, len(cases))
        evaluate(
            test_cases=cases,
            metrics=metrics,
            # Group by strategy so results are distinguishable in output
            experiment_name=f"mongo_retrieval_{name}",
        )


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MongoDB RAG retrievers with DeepEval")
    parser.add_argument("--mongo-uri", default="mongodb://admin:password@localhost:27017")
    parser.add_argument("--database", default="rag")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--vector-index", default="rag_vector_index")
    parser.add_argument("--text-index", default="rag_text_index")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--vllm-model", default="qwen25-coder-32b-awq")
    parser.add_argument("--vllm-base-url", default="http://192.168.125.31:8000/v1")
    parser.add_argument("--golden-path", type=Path, default=Path("data/eval/golden_retrieval.json"))
    parser.add_argument("--generate-golden", action="store_true", help="Regenerate golden dataset with synthesizer")
    parser.add_argument("--num-golden-questions", type=int, default=50)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    collection = connect_mongo(args.mongo_uri, args.database, args.collection)

    if args.generate_golden or not args.golden_path.exists():
        golden_cases = generate_golden_dataset(
            collection=collection,
            output_path=args.golden_path,
            vllm_model=args.vllm_model,
            vllm_base_url=args.vllm_base_url,
            num_docs_sample=1000,
            max_docs_for_corpus=100,
            num_questions=args.num_golden_questions,
        )
    else:
        logging.info("Loading existing golden dataset from %s", args.golden_path)
        golden_dataset = Golden()
        golden_dataset.from_json(str(args.golden_path))
        golden_cases = golden_dataset.test_cases

    evaluate_retrievers(
        collection=collection,
        vector_index=args.vector_index,
        text_index=args.text_index,
        embedding_dim=args.embedding_dim,
        golden_cases=golden_cases,
        top_k=5,
    )


if __name__ == "__main__":
    main()
