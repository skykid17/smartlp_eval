#!/usr/bin/env python3
"""Standalone MongoDB RAG workflow with ingestion, hybrid retrieval, and CLI orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4
from urllib.parse import quote_plus

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".pdf"}
DEFAULT_ALLOWED_METADATA = ["source", "category", "tags", "file_type", "collection"]
DEFAULT_TEXT_PATHS = ["content", "metadata.source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MongoDB Community RAG toolkit")
    parser.add_argument("mode", choices=["init", "ingest", "query", "test"], help="Workflow stage to run")
    parser.add_argument("--mongo-uri", default=f"mongodb://admin:{quote_plus("P@55w0rd")}@localhost:27018", help="MongoDB connection URI")
    parser.add_argument("--database", default="rag", help="Database name")
    parser.add_argument("--collection", default="documents", help="Collection for chunks")
    parser.add_argument("--vector-index", default="rag_vector_index", help="Vector search index name")
    parser.add_argument("--text-index", default="rag_text_index", help="Text search index name")
    parser.add_argument("--text-paths", nargs="*", default=DEFAULT_TEXT_PATHS, help="Fields included in the text index")
    parser.add_argument("--input-path", type=Path, help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128, help="Insert batch size")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--embedding-provider", default="placeholder-provider", help="Label stored with vectors")
    parser.add_argument("--allowed-metadata", nargs="*", default=DEFAULT_ALLOWED_METADATA, help="Metadata fields kept before insert")
    parser.add_argument("--query-text", help="User query for query mode")
    parser.add_argument("--top-k", type=int, default=5, help="Results returned to the chain")
    parser.add_argument("--semantic-candidates", type=int, default=50, help="Candidate pool for vector search")
    parser.add_argument("--keyword-candidates", type=int, default=30, help="Candidate pool for keyword search")
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal Rank Fusion constant")
    parser.add_argument("--disable-native-rankfusion", action="store_true", help="Force Python-based RRF even if $rankFusion exists")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", help="Skip writes during ingest for validation")
    parser.add_argument("--text-language", default="english", help="Language for the text index")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")


def connect(uri: str) -> MongoClient:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except ServerSelectionTimeoutError as exc:
        logging.error("Unable to reach MongoDB at %s: %s", uri, exc)
        raise


def ensure_vector_index(collection: Collection, index_name: str, embedding_dim: int) -> None:
    definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "type": "vector",
                    "similarity": "cosine",
                    "dimensions": embedding_dim,
                },
                "content": {"type": "string"},
                "metadata": {"type": "document"},
            },
        }
    }
    try:
        existing = collection.database.command(
            {
                "listSearchIndexes": collection.name,
                "name": index_name,
            }
        )
        if any(idx.get("name") == index_name for idx in existing.get("indexes", [])):
            logging.info("Vector search index '%s' already exists", index_name)
            return
    except OperationFailure as exc:
        code = getattr(exc, "code", None)
        if code == 31082 or "SearchNotEnabled" in str(exc):
            logging.warning(
                "$listSearchIndexes unavailable on this deployment; assuming index '%s' is missing",
                index_name,
            )
        elif "NamespaceNotFound" in str(exc):
            pass
        else:
            raise
    payload = {
        "createSearchIndexes": collection.name,
        "indexes": [
            {
                "name": index_name,
                "definition": definition,
            }
        ],
    }
    logging.info("Creating vector search index '%s'", index_name)
    try:
        collection.database.command(payload)
    except OperationFailure as exc:
        if "already exists" in str(exc):
            logging.info("Vector search index '%s' already exists (reported by server)", index_name)
            return
        raise


def ensure_text_index(collection: Collection, index_name: str, text_paths: Sequence[str], language: str) -> None:
    existing = collection.index_information()
    if index_name in existing:
        logging.info("Text index '%s' already exists", index_name)
        return
    index_fields = [ (path, "text") for path in text_paths ]
    logging.info("Creating text index '%s' on %s", index_name, text_paths)
    collection.create_index(index_fields, name=index_name, default_language=language)


def load_documents(input_path: Path) -> List[Document]:
    from langchain_community.document_loaders import JSONLoader, PyPDFLoader, TextLoader

    def load_file(path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logging.debug("Skipping unsupported file %s", path)
            return []
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        elif suffix in {".yaml", ".yml"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        elif suffix == ".json":
            loader = JSONLoader(str(path), jq_schema=".", text_content=False)
            docs = loader.load()
        elif suffix == ".csv":
            content = path.read_text(encoding="utf-8")
            docs = [Document(page_content=content, metadata={})]
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        else:
            docs = []
        for doc in docs:
            doc.metadata.setdefault("source", path.name)
            doc.metadata.setdefault("file_path", str(path))
            doc.metadata.setdefault("file_type", suffix.lstrip("."))
        return docs

    path = input_path
    if path.is_file():
        return load_file(path)
    docs: List[Document] = []
    for file_path in path.rglob("*"):
        if file_path.is_file():
            docs.extend(load_file(file_path))
    return docs


def filter_metadata(metadata: Dict, allowed_fields: Sequence[str]) -> Dict:
    allowed = set(allowed_fields)
    filtered = {k: v for k, v in metadata.items() if k in allowed and v not in (None, "")}
    return filtered


def generate_embeddings(texts: Sequence[str], dim: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        while len(values) < dim:
            for byte in digest:
                values.append((byte / 255.0) - 0.5)
                if len(values) == dim:
                    break
        vectors.append(values)
    return vectors


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def chunk_to_record(chunk: Document, embedding: List[float], provider: str, allowed_metadata: Sequence[str]) -> Dict:
    metadata = filter_metadata(dict(chunk.metadata or {}), allowed_metadata)
    metadata.setdefault("source", chunk.metadata.get("source", "unknown"))
    metadata.setdefault("file_type", chunk.metadata.get("file_type", "text"))
    return {
        "chunk_id": str(uuid4()),
        "content": chunk.page_content,
        "metadata": metadata,
        "embedding": embedding,
        "embedding_provider": provider,
        "created_at": datetime.utcnow(),
        "hash": hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest(),
    }


def ingest_documents(args: argparse.Namespace, collection: Collection) -> None:
    if not args.input_path:
        logging.error("--input-path is required for ingest mode")
        sys.exit(1)
    docs = load_documents(args.input_path)
    if not docs:
        logging.warning("No documents found under %s", args.input_path)
        return
    logging.info("Loaded %d base documents", len(docs))
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    if not chunks:
        logging.warning("No chunks produced; check chunk parameters")
        return
    logging.info("Generated %d chunks", len(chunks))
    embeddings = generate_embeddings([chunk.page_content for chunk in chunks], args.embedding_dim)
    batch: List[Dict] = []
    inserted = 0
    for chunk, embedding in zip(chunks, embeddings):
        batch.append(chunk_to_record(chunk, embedding, args.embedding_provider, args.allowed_metadata))
        if len(batch) >= args.batch_size:
            if not args.dry_run:
                collection.insert_many(batch, ordered=False)
            inserted += len(batch)
            batch.clear()
    if batch and not args.dry_run:
        collection.insert_many(batch, ordered=False)
        inserted += len(batch)
    logging.info("Ingest complete. %d chunks %s", inserted, "simulated" if args.dry_run else "inserted")


def reciprocal_rank_fusion(runs: Sequence[List[Dict]], k: int, limit: int) -> List[Dict]:
    scores: Dict[str, float] = defaultdict(float)
    docs: Dict[str, Dict] = {}
    for run in runs:
        for rank, doc in enumerate(run):
            doc_id = str(doc["_id"])
            docs[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in ranked[:limit]]


def format_docs(docs: Sequence[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{idx}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


class MongoHybridRetriever(BaseRetriever):
    def __init__(
        self,
        collection: Collection,
        embedding_dim: int,
        vector_index: str,
        text_index: str,
        top_k: int,
        semantic_candidates: int,
        keyword_candidates: int,
        rrf_k: int,
        allowed_text_paths: Sequence[str],
        native_rankfusion: bool = True,
    ) -> None:
        super().__init__()
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.vector_index = vector_index
        self.text_index = text_index
        self.top_k = top_k
        self.semantic_candidates = semantic_candidates
        self.keyword_candidates = keyword_candidates
        self.rrf_k = rrf_k
        self.allowed_text_paths = allowed_text_paths
        self.native_rankfusion = native_rankfusion

    def _get_relevant_documents(self, query: str, *, run_native: bool = True) -> List[Document]:
        query_vector = generate_embeddings([query], self.embedding_dim)[0]
        if self.native_rankfusion and run_native:
            try:
                docs = self._native_rank_fusion(query, query_vector)
                if docs:
                    return docs
            except OperationFailure as exc:
                if "Unrecognized pipeline stage" in str(exc):
                    logging.warning("$rankFusion not available; falling back to manual RRF")
                else:
                    logging.error("$rankFusion error: %s", exc)
                self.native_rankfusion = False
        manual_docs = self._manual_rrf(query, query_vector)
        return manual_docs

    def _native_rank_fusion(self, query: str, query_vector: List[float]) -> List[Document]:
        pipeline = [
            {
                "$rankFusion": {
                    "indexes": [
                        {
                            "type": "vectorSearch",
                            "index": self.vector_index,
                            "path": "embedding",
                            "queryVector": query_vector,
                            "numCandidates": self.semantic_candidates,
                            "limit": self.top_k,
                        },
                        {
                            "type": "search",
                            "index": self.text_index,
                            "text": {
                                "path": list(self.allowed_text_paths),
                                "query": query,
                            },
                            "limit": self.top_k,
                        },
                    ],
                    "rrf": {"k": self.rrf_k},
                    "limit": self.top_k,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "fusionScore"},
                }
            },
        ]
        docs = list(self.collection.aggregate(pipeline))
        return [Document(page_content=doc.get("content", ""), metadata={**doc.get("metadata", {}), "score": doc.get("score")}) for doc in docs]

    def _manual_rrf(self, query: str, query_vector: List[float]) -> List[Document]:
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": self.semantic_candidates,
                    "limit": self.top_k,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        text_pipeline = [
            {
                "$search": {
                    "index": self.text_index,
                    "text": {
                        "query": query,
                        "path": list(self.allowed_text_paths),
                    },
                }
            },
            {"$limit": self.keyword_candidates},
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]
        vector_results = list(self.collection.aggregate(vector_pipeline))
        text_results = list(self.collection.aggregate(text_pipeline))
        fused = reciprocal_rank_fusion([vector_results, text_results], self.rrf_k, self.top_k)
        return [Document(page_content=doc.get("content", ""), metadata={**doc.get("metadata", {}), "score": doc.get("score")}) for doc in fused]

    def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async usage not implemented; use sync retriever")

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def build_chain(retriever: MongoHybridRetriever) -> RunnableLambda:
    prompt = PromptTemplate(
        template=(
            "You are a SOC assistant. Use the context to answer the question.\n"
            "Question: {question}\nContext:\n{context}\nAnswer:"
        ),
        input_variables=["question", "context"],
    )

    def mock_llm(text: str) -> Dict[str, str]:
        return {"answer": f"[MOCK COMPLETION]\n{text}"}

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(
                lambda question: format_docs(retriever.get_relevant_documents(question))
            ),
        }
        | prompt
        | RunnableLambda(mock_llm)
    )
    return chain


def run_query(args: argparse.Namespace, collection: Collection) -> None:
    if not args.query_text:
        logging.error("--query-text is required for query mode")
        sys.exit(1)
    retriever = MongoHybridRetriever(
        collection=collection,
        embedding_dim=args.embedding_dim,
        vector_index=args.vector_index,
        text_index=args.text_index,
        top_k=args.top_k,
        semantic_candidates=args.semantic_candidates,
        keyword_candidates=args.keyword_candidates,
        rrf_k=args.rrf_k,
        allowed_text_paths=args.text_paths,
        native_rankfusion=not args.disable_native_rankfusion,
    )
    chain = build_chain(retriever)
    result = chain.invoke(args.query_text)
    print(json.dumps(result, indent=2, default=str))


def run_test(args: argparse.Namespace, collection: Collection) -> None:
    zero_vector = [0.0] * args.embedding_dim
    pipeline = [
        {
            "$vectorSearch": {
                "index": args.vector_index,
                "path": "embedding",
                "queryVector": zero_vector,
                "numCandidates": 5,
                "limit": 1,
            }
        },
        {"$limit": 1},
    ]
    try:
        list(collection.aggregate(pipeline))
        logging.info("Vector search pipeline executed. mongot is reachable and index '%s' responded.", args.vector_index)
    except OperationFailure as exc:
        logging.error("Vector search test failed: %s", exc)
        raise


def run_init(args: argparse.Namespace, collection: Collection) -> None:
    ensure_text_index(collection, args.text_index, args.text_paths, args.text_language)
    ensure_vector_index(collection, args.vector_index, args.embedding_dim)
    logging.info("Initialization complete")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    client = connect(args.mongo_uri)
    collection = client[args.database][args.collection]

    if args.mode == "init":
        run_init(args, collection)
    elif args.mode == "ingest":
        run_init(args, collection)
        ingest_documents(args, collection)
    elif args.mode == "query":
        run_query(args, collection)
    elif args.mode == "test":
        run_test(args, collection)
    else:
        logging.error("Unsupported mode %s", args.mode)


if __name__ == "__main__":
    main()
