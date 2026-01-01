#!/usr/bin/env python3
"""Refactored MongoDB RAG toolkit with Python-only fallback retriever.

This is the patched module that:
- Preserves your original behavior (attempt Atlas/mongot searches)
- Adds a robust Python fallback retriever using SentenceTransformers + cosine sim + keyword scoring
- Ensures retrieval returns documents even on MongoDB Community / no mongot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, OperationFailure, ServerSelectionTimeoutError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# --- Defaults ---
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".pdf"}
DEFAULT_ALLOWED_METADATA = ["source", "category", "tags", "file_type", "collection"]
DEFAULT_TEXT_PATHS = ["content", "metadata.source"]

# --- Utility helpers ---


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def filter_metadata(metadata: Dict, allowed_fields: Sequence[str]) -> Dict:
    allowed = set(allowed_fields)
    return {k: v for k, v in metadata.items() if k in allowed and v not in (None, "")}


# --- RRF fusion ---


def reciprocal_rank_fusion(runs: Sequence[List[Dict]], k: int, limit: int) -> List[Dict]:
    scores: Dict[str, float] = defaultdict(float)
    docs: Dict[str, Dict] = {}
    for run in runs:
        for rank, doc in enumerate(run):
            doc_id = str(doc.get("_id")) if doc.get("_id") is not None else str(hash(json.dumps(doc, sort_keys=True)))
            docs[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in ranked[:limit]]


def format_docs(docs: Sequence[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] Source: {d.metadata.get('source','unknown')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# --- RAG class (single programmatic entrypoint) ---


class RAG:
    def __init__(
        self,
        mongo_uri: str = "mongodb://admin:password@localhost:27017",
        database: str = "soc_rag_db",
        collection_name: str = "knowledge_base",
        embedding_dim: int = 384,
        embedding_provider: str = "all-MiniLM-L6-v2",
        vector_index: str = "rag_vector_index",
        text_index: str = "rag_text_index",
        text_paths: Sequence[str] = DEFAULT_TEXT_PATHS,
        text_language: str = "english",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_batch_size: int = 512,
        batch_size: int = 128,
    ) -> None:
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.text_index = text_index
        self.text_paths = list(text_paths)
        self.text_language = text_language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size
        self.batch_size = batch_size

        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    # --- Connection / index helpers ---
    def connect(self) -> MongoClient:
        if self.client is None:
            try:
                client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                self.client = client
            except ServerSelectionTimeoutError as exc:
                logging.error("Unable to reach MongoDB at %s: %s", self.mongo_uri, exc)
                raise
        return self.client

    def _ensure_collection(self) -> Collection:
        if self.collection is None:
            client = self.connect()
            self.collection = client[self.database][self.collection_name]
        return self.collection

    def init(self) -> None:
        coll = self._ensure_collection()
        ensure_text_index(coll, self.text_index, self.text_paths, self.text_language)
        ensure_vector_index(coll, self.vector_index, self.embedding_dim)
        logging.info("RAG initialization complete")

    # --- Embeddings ---
    def get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logging.info("Loading SentenceTransformer: %s", self.embedding_provider)
            self._embedding_model = SentenceTransformer(self.embedding_provider)
        return self._embedding_model

    def generate_embeddings(self, texts: Sequence[str], show_progress: bool = False) -> List[List[float]]:
        embedder = self.get_embedding_model()
        embeddings = embedder.encode(texts, show_progress_bar=show_progress)
        # embeddings may be numpy array — convert to list of lists
        return np.asarray(embeddings).tolist()

    # --- Document loading / chunking ---
    def load_documents(self, input_path: Path) -> List[Document]:
        from langchain_community.document_loaders import JSONLoader, PyPDFLoader, TextLoader

        def load_file(path: Path) -> List[Document]:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logging.debug("Skipping unsupported file %s", path)
                return []
            if suffix in {".txt", ".md", ".yaml", ".yml"}:
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

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(docs)

    def chunk_to_record(self, chunk: Document, embedding: List[float], provider: str, allowed_metadata: Sequence[str]) -> Dict:
        metadata = filter_metadata(dict(chunk.metadata or {}), allowed_metadata)
        metadata.setdefault("source", chunk.metadata.get("source", "unknown"))
        metadata.setdefault("file_type", chunk.metadata.get("file_type", "text"))
        content_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        return {
            "_id": content_hash,
            "chunk_id": content_hash,
            "content": chunk.page_content,
            "metadata": metadata,
            "embedding": embedding,
            "embedding_provider": provider,
            "created_at": datetime.utcnow(),
            "hash": content_hash,
        }

    # --- Ingest ---
    def ingest(self, input_path: Path, category: Optional[str] = None, dry_run: bool = False, allowed_metadata: Sequence[str] = DEFAULT_ALLOWED_METADATA) -> int:
        coll = self._ensure_collection()
        docs = self.load_documents(input_path)
        if not docs:
            logging.warning("No documents found under %s", input_path)
            return 0
        chunks = self.chunk_documents(docs)
        if not chunks:
            logging.warning("No chunks produced; check chunk parameters")
            return 0
        logging.info("Loaded %d docs -> %d chunks", len(docs), len(chunks))

        # Delta check
        content_hashes = [hashlib.sha1(c.page_content.encode("utf-8")).hexdigest() for c in chunks]
        existing_ids: set[str] = set()
        batch_size = 5000
        for i in range(0, len(content_hashes), batch_size):
            batch = content_hashes[i: i + batch_size]
            if not batch:
                continue
            for doc in coll.find({"_id": {"$in": batch}}, {"_id": 1}):
                existing_ids.add(doc["_id"])
        chunks_to_embed = [chunk for chunk, h in zip(chunks, content_hashes) if h not in existing_ids]
        if not chunks_to_embed:
            logging.info("All %d chunks already ingested; skipping embedding generation", len(chunks))
            return 0

        mongo_batch: List[Dict] = []
        inserted = 0

        for chunk_batch in batched(chunks_to_embed, self.embedding_batch_size):
            texts = [c.page_content for c in chunk_batch]
            embeddings = self.generate_embeddings(texts, show_progress=False)
            for chunk, embedding in zip(chunk_batch, embeddings):
                if category:
                    chunk.metadata["category"] = category
                mongo_batch.append(self.chunk_to_record(chunk, embedding, self.embedding_provider, allowed_metadata))
                if len(mongo_batch) >= self.batch_size:
                    if not dry_run:
                        try:
                            res = coll.insert_many(mongo_batch, ordered=False)
                            inserted += len(res.inserted_ids)
                        except BulkWriteError as exc:
                            dup_errors = [err for err in exc.details.get("writeErrors", []) if err.get("code") == 11000]
                            inserted += exc.details.get("nInserted", 0)
                            if dup_errors:
                                logging.info("Skipped %d duplicate chunks during insert", len(dup_errors))
                            non_dup = [err for err in exc.details.get("writeErrors", []) if err.get("code") != 11000]
                            if non_dup:
                                raise
                    else:
                        inserted += len(mongo_batch)
                    mongo_batch.clear()
        if mongo_batch:
            if not dry_run:
                try:
                    res = coll.insert_many(mongo_batch, ordered=False)
                    inserted += len(res.inserted_ids)
                except BulkWriteError as exc:
                    dup_errors = [err for err in exc.details.get("writeErrors", []) if err.get("code") == 11000]
                    inserted += exc.details.get("nInserted", 0)
                    if dup_errors:
                        logging.info("Skipped %d duplicate chunks during insert", len(dup_errors))
                    non_dup = [err for err in exc.details.get("writeErrors", []) if err.get("code") != 11000]
                    if non_dup:
                        raise
            else:
                inserted += len(mongo_batch)
        logging.info("Ingest complete. %d chunks %s", inserted, "simulated" if dry_run else "inserted")
        return inserted

    # --- Python-only fallback: semantic + keyword + RRF ---
    def _py_fallback_retrieve(
        self,
        query: str,
        limit: int = 5,
        semantic_k: int = 50,
        keyword_k: int = 50,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Python-only hybrid retriever (no mongot, no vector search).
        Uses local embeddings + cosine similarity + keyword scoring, fused by RRF.
        """
        coll = self._ensure_collection()

        # Load docs (project minimal fields)
        all_docs = list(coll.find({}, {"content": 1, "metadata": 1, "embedding": 1, "_id": 1}).limit(10000))
        if not all_docs:
            return []

        # Compute query embedding
        q_emb = self.generate_embeddings([query], show_progress=False)[0]

        # Cosine similarity
        def cosine(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(a.dot(b) / denom) if denom > 0 else 0.0

        for doc in all_docs:
            emb = doc.get("embedding")
            if emb:
                try:
                    doc["_sim_score"] = cosine(q_emb, emb)
                except Exception:
                    doc["_sim_score"] = 0.0
            else:
                doc["_sim_score"] = 0.0

        semantic_sorted = sorted(all_docs, key=lambda x: x.get("_sim_score", 0.0), reverse=True)
        semantic_top = semantic_sorted[:semantic_k]

        # Keyword scoring — token overlap
        tokens = set(re.findall(r"\w+", query.lower()))
        for doc in all_docs:
            text = (doc.get("content") or "").lower()
            # simple presence count
            count = sum(1 for t in tokens if t in text)
            doc["_kw_score"] = count

        keyword_sorted = sorted(all_docs, key=lambda x: x.get("_kw_score", 0), reverse=True)
        keyword_top = keyword_sorted[:keyword_k]

        # Convert to the same dict shape expected by RRF: with possibly _id, content, metadata, score
        sem_run = [
            {"_id": d.get("_id"), "content": d.get("content", ""), "metadata": d.get("metadata", {}), "score": d.get("_sim_score")}
            for d in semantic_top
        ]
        kw_run = [
            {"_id": d.get("_id"), "content": d.get("content", ""), "metadata": d.get("metadata", {}), "score": d.get("_kw_score")}
            for d in keyword_top
        ]

        fused = reciprocal_rank_fusion([sem_run, kw_run], rrf_k, limit)
        return [Document(page_content=d.get("content", ""), metadata=d.get("metadata", {})) for d in fused]

    # --- Retriever ---
    class _MongoHybridRetriever:
        def __init__(
            self,
            collection: Collection,
            embedding_fn,
            embedding_dim: int,
            vector_index: str,
            text_index: str,
            top_k: int,
            semantic_candidates: int,
            keyword_candidates: int,
            rrf_k: int,
            allowed_text_paths: Sequence[str],
            filter_category: Optional[str] = None,
        ) -> None:
            self.collection = collection
            self.embedding_fn = embedding_fn
            self.embedding_dim = embedding_dim
            self.vector_index = vector_index
            self.text_index = text_index
            self.top_k = top_k
            self.semantic_candidates = semantic_candidates
            self.keyword_candidates = keyword_candidates
            self.rrf_k = rrf_k
            self.allowed_text_paths = allowed_text_paths
            self.filter_category = filter_category
            # parent will be set by RAG.query so fallback can call back
            self.parent: Optional["RAG"] = None

        def invoke(self, query: str) -> List[Document]:
            qv = self.embedding_fn([query])[0]
            return self._manual_rrf(query, qv)

        def _get_relevant_documents(self, query: str, *, run_native: bool = False) -> List[Document]:
            qv = self.embedding_fn([query])[0]
            return self._manual_rrf(query, qv)

        def get_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)

        def _manual_rrf(self, query: str, query_vector: List[float]) -> List[Document]:
            vector_results: List[Dict] = []
            text_results: List[Dict] = []

            # Vector search (Atlas vector search / $vectorSearch) - only run if semantic candidates configured
            if getattr(self, "semantic_candidates", 0) and self.semantic_candidates > 0:
                vector_search_spec = {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": self.semantic_candidates,
                    "limit": self.top_k,
                }
                if self.filter_category:
                    vector_search_spec["filter"] = {"metadata.category": {"$eq": self.filter_category}}

                vector_pipeline = [
                    {"$vectorSearch": vector_search_spec},
                    {"$project": {"content": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}},
                    {"$unset": "embedding"},
                ]
                try:
                    vector_results = list(self.collection.aggregate(vector_pipeline, allowDiskUse=True))
                except OperationFailure as exc:
                    logging.warning("Vector search unavailable or failed; continuing without vector results: %s", exc)
                    vector_results = []
                except Exception as exc:
                    logging.warning("Vector search raised error; continuing: %s", exc)
                    vector_results = []

            # Text search ($search) with fallback to $text for non-Atlas deployments
            if getattr(self, "keyword_candidates", 0) and self.keyword_candidates > 0:
                text_pipeline = [
                    {"$search": {"index": self.text_index, "text": {"query": query, "path": list(self.allowed_text_paths)}}},
                    {"$limit": self.keyword_candidates},
                    {"$project": {"content": 1, "metadata": 1, "score": {"$meta": "searchScore"}}},
                    {"$unset": "embedding"},
                ]
                try:
                    text_results = list(self.collection.aggregate(text_pipeline, allowDiskUse=True))
                except OperationFailure as exc:
                    logging.warning("$search not available; falling back to $text: %s", exc)
                    text_results = []
                    try:
                        text_filter: Dict = {"$text": {"$search": query}}
                        if self.filter_category:
                            text_filter["metadata.category"] = {"$eq": self.filter_category}
                        fallback_cursor = (
                            self.collection.find(text_filter, {"content": 1, "metadata": 1, "score": {"$meta": "textScore"}})
                            .limit(self.keyword_candidates)
                        )
                        text_results = list(fallback_cursor)
                    except Exception:
                        logging.warning("Fallback $text query failed; continuing without text results")
                        text_results = []
                except Exception as exc:
                    logging.warning("Text search raised error; continuing: %s", exc)
                    text_results = []

            # If both vector and text searches returned nothing, fallback to Python-only retriever
            if not vector_results and not text_results:
                if self.parent is not None:
                    logging.info("No MongoDB search results — using Python fallback retriever")
                    return self.parent._py_fallback_retrieve(
                        query=query,
                        limit=self.top_k,
                        semantic_k=self.semantic_candidates,
                        keyword_k=self.keyword_candidates,
                        rrf_k=self.rrf_k,
                    )
                else:
                    logging.warning("No parent configured for fallback retriever; returning empty list")
                    return []

            # Otherwise fuse results using RRF
            fused = reciprocal_rank_fusion([vector_results, text_results], self.rrf_k, self.top_k)
            return [
                Document(page_content=doc.get("content", ""), metadata={**doc.get("metadata", {}), "score": doc.get("score")})
                for doc in fused
            ]

        def aget_relevant_documents(self, query: str) -> List[Document]:
            raise NotImplementedError("Async usage not implemented; use sync invoke()")

    # --- Chain builder (kept simple) ---
    def _build_chain(self, retriever: _MongoHybridRetriever) -> RunnableLambda:
        # Example: using local LLM endpoint; replace as needed
        llm = ChatOpenAI(model="qwen25-coder-32b-awq", base_url="http://192.168.125.31:8000/v1", api_key="test", temperature=0)
        prompt = PromptTemplate(
            template=(
                "You are a SOC assistant. Use the context to answer the question.\n"
                "Question: {question}\nContext:\n{context}\nAnswer:"
            ),
            input_variables=["question", "context"],
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(lambda question: format_docs(retriever.invoke(question))),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    # --- Public query API ---
    def query(
        self,
        question: str,
        top_k: int = 5,
        semantic_candidates: int = 50,
        keyword_candidates: int = 30,
        rrf_k: int = 60,
        allowed_text_paths: Sequence[str] = DEFAULT_TEXT_PATHS,
        filter_category: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        coll = self._ensure_collection()
        retriever = self._MongoHybridRetriever(
            collection=coll,
            embedding_fn=lambda texts: self.generate_embeddings(texts, show_progress=False),
            embedding_dim=self.embedding_dim,
            vector_index=self.vector_index,
            text_index=self.text_index,
            top_k=top_k,
            semantic_candidates=semantic_candidates,
            keyword_candidates=keyword_candidates,
            rrf_k=rrf_k,
            allowed_text_paths=list(allowed_text_paths),
            filter_category=filter_category,
        )
        # attach parent so the retriever can call the python fallback
        retriever.parent = self

        docs = retriever.invoke(question)
        chain = self._build_chain(retriever)
        answer = chain.invoke(question)
        retrieval_context = [
            {"content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        return answer, retrieval_context


# --- Index helpers (reused) ---


def ensure_vector_index(collection: Collection, index_name: str, embedding_dim: int) -> None:
    definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {"type": "knnVector", "similarity": "cosine", "dimensions": embedding_dim},
                "content": {"type": "string"},
                "metadata": {"type": "document", "fields": {"category": {"type": "token"}}},
            },
        }
    }
    try:
        existing = collection.database.command({"listSearchIndexes": collection.name, "name": index_name})
        if any(idx.get("name") == index_name for idx in existing.get("indexes", [])):
            logging.info("Vector search index '%s' already exists", index_name)
            return
    except OperationFailure as exc:
        code = getattr(exc, "code", None)
        if code == 31082 or "SearchNotEnabled" in str(exc):
            logging.warning("$listSearchIndexes unavailable on this deployment; assuming index '%s' is missing", index_name)
        elif "NamespaceNotFound" in str(exc):
            pass
        else:
            raise

    payload = {"createSearchIndexes": collection.name, "indexes": [{"name": index_name, "definition": definition}]}
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
    index_fields = [(path, "text") for path in text_paths]
    logging.info("Creating text index '%s' on %s", index_name, text_paths)
    collection.create_index(index_fields, name=index_name, default_language=language)


# --- CLI wrapper (thin) ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored RAG CLI")
    parser.add_argument("mode", choices=["init", "ingest", "query", "test"], help="operation")
    parser.add_argument("--mongo-uri", default="mongodb://admin:password@localhost:27017")
    parser.add_argument("--database", default="rag")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--query-text", default="Which package/add on do I install to parse windows_xml logs into elastic? Return only the name of the package/add on.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--category")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    rag = RAG(mongo_uri=args.mongo_uri, database=args.database, collection_name=args.collection)

    if args.mode == "init":
        rag.init()
    elif args.mode == "ingest":
        if not args.input_path:
            logging.error("--input-path required for ingest")
            sys.exit(1)
        rag.init()
        rag.ingest(args.input_path, category=args.category, dry_run=args.dry_run)
    elif args.mode == "query":
        rag.init()
        answer, ctx = rag.query(args.query_text)
        print(json.dumps({"answer": answer, "retrieval_context": ctx}, indent=2, default=str))
    elif args.mode == "test":
        coll = rag._ensure_collection()
        try:
            qv = rag.generate_embeddings(["Smoke test"], show_progress=False)[0]
        except Exception as exc:
            logging.error("Failed to generate test embedding: %s", exc)
            raise
        if not any(x != 0.0 for x in qv):
            logging.error("Generated test embedding is all zeros; aborting vector test")
            raise RuntimeError("test embedding is zero vector")
        pipeline = [{"$vectorSearch": {"index": rag.vector_index, "path": "embedding", "queryVector": qv, "numCandidates": 5, "limit": 1}}, {"$limit": 1}]
        try:
            list(coll.aggregate(pipeline))
            logging.info("Vector search pipeline executed. mongot is reachable and index '%s' responded.", rag.vector_index)
        except OperationFailure as exc:
            logging.error("Vector search test failed: %s", exc)
            raise


if __name__ == "__main__":
    main()
