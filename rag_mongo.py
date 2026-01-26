#!/usr/bin/env python3
"""MongoDB RAG toolkit with Python-only fallback retriever.

This version is updated to support:
- MongoDB Atlas Local (Docker) with 'vectorSearch' type indexes.
- Direct connection URI handling for local Docker setups.
- Renamed indexes (vector_index, text_index).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from enum import Enum
import time
from dataclasses import dataclass
from tqdm import tqdm 
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Any
import spacy

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, OperationFailure, ServerSelectionTimeoutError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

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


def format_docs(docs: Sequence[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] Source: {d.metadata.get('source','unknown')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )

def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class ScoredDoc:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float

class RetrievalMode(str, Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


DOMAIN_TERMS = {"powershell","elasticsearch","splunk","elastic","log4j","base64","kibana","windows","logstash","wazuh","mitre","siem"}

_nlp = None
def get_nlp(): # Lazy load spaCy model
    global _nlp 
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_keywords(text: str, max_keywords: int = 15):
        doc = get_nlp()(text.lower())

        keywords = []
        seen = set()

        for token in doc:
            # Skip stopwords, punctuation, spaces, numbers
            if token.is_stop or token.is_punct or token.like_num:
                continue

            lemma = token.lemma_.strip()

            # Skip empty strings & 1-char words
            if not lemma or len(lemma) < 2:
                continue

            # Keep nouns + adjectives + domain terms
            if (
                token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
                or lemma in DOMAIN_TERMS
            ):
                if lemma not in seen:
                    seen.add(lemma)
                    keywords.append(lemma)

            # Stop when we reach the limit
            if len(keywords) >= max_keywords:
                break

        return " ".join(keywords) if keywords else ""

# --- Retriever Class ---
class MongoHybridRetriever:
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
        fallback_engine: Optional[LocalRetriever] = None,
        filter_category: Optional[str] = None
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
        self.fallback_engine = fallback_engine
        self.filter_category = filter_category

    def invoke(self, query: str) -> List[Document]:
        query_vector = self.embedding_fn([query])[0]
        query_text = extract_keywords(query)

        # --- Primary Path: Mongo RankFusion ---
        try:
            vector_pipeline = [
                {"$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": self.semantic_candidates,
                    "limit": self.top_k,
                    **(
                        {"filter": {"metadata.category": self.filter_category}}
                        if self.filter_category else {}
                    )
                }}
            ]

            text_pipeline = [
                {"$search": {
                    "index": self.text_index,
                    "phrase": {"query": query_text, "path": "content"}
                }},
                {"$limit": self.keyword_candidates}
            ]

            pipeline = [
                {"$rankFusion": {
                    "input": {"pipelines": {
                        "vectorPipeline": vector_pipeline,
                        "fullTextPipeline": text_pipeline
                    }},
                    "combination": {"weights": {
                        "vectorPipeline": 0.5,
                        "fullTextPipeline": 0.5
                    }},
                    "scoreDetails": True
                }},
                {"$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "scoreDetails": {"$meta": "scoreDetails"}
                }},
                {"$limit": self.top_k}
            ]

            fused = list(self.collection.aggregate(pipeline))

            if fused:
                return [
                    Document(
                        page_content=d.get("content", ""),
                        metadata={
                            **d.get("metadata", {}),
                            "scoreDetails": d.get("scoreDetails")
                        }
                    )
                    for d in fused
                ]

        except Exception as exc:
            logger.error("RankFusion failed: %s", exc)

        return []


class LocalRetriever:
    """
    Python-only fallback retriever.

    Used when MongoDB Atlas RankFusion or vectorSearch is unavailable.
    Performs:
    - Candidate selection via MongoDB
    - In-memory vector similarity
    - Keyword search
    - Reciprocal Rank Fusion
    """

    def __init__(
        self,
        collection: Collection,
        embedding_fn,
        text_index: str,
        text_paths: Sequence[str] = DEFAULT_TEXT_PATHS,
    ):
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.text_index = text_index
        self.text_paths = list(text_paths)

    # -------------------------
    # Public API
    # -------------------------
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        limit: int = 5,
        **kwargs,
    ) -> List[Document]:

        base_filter = self._build_base_filter(kwargs.get("filter_category"))
        keyword_query = extract_keywords(query)
        candidates = self._fetch_candidates(base_filter, keyword_query)

        if not candidates:
            return []

        if mode == RetrievalMode.VECTOR:
            results = self._semantic_search(query, candidates, limit)

        elif mode == RetrievalMode.TEXT:
            results = self._keyword_search(keyword_query, limit, base_filter)

        else:  # HYBRID
            sem = self._semantic_search(query, candidates, kwargs.get("semantic_k", 50))
            txt = self._keyword_search(keyword_query, kwargs.get("keyword_k", 50), base_filter)
            results = self._rrf_fuse([sem, txt], k=kwargs.get("rrf_k", 60), limit=limit)

        return self._to_documents(results)

    def _build_base_filter(self, category: Optional[str]) -> Dict:
        base_filter = {}
        if category:
            base_filter["metadata.category"] = category
        return base_filter
    
    def _to_documents(self, scored_docs: List[ScoredDoc]) -> List[Document]:
        return [
            Document(
                page_content=d.content,
                metadata={**d.metadata, "score": d.score}
            )
            for d in scored_docs
        ]

    def _fetch_candidates(
        self,
        base_filter: Dict,
        keyword_query: str,
        max_docs: int = 5000,
    ) -> List[Dict]:

        query = dict(base_filter)
        if keyword_query:
            query["$text"] = {"$search": keyword_query}

        return list(
            self.collection.find(
                query,
                {"content": 1, "metadata": 1, "embedding": 1},
            ).limit(max_docs)
        )

    # Helper: Vector Search
    def _semantic_search(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[ScoredDoc]:

        query_vec = self.embedding_fn([query], show_progress=False)[0]

        scored = []
        for doc in candidates:
            emb = doc.get("embedding")
            if not emb:
                continue

            score = cosine(query_vec, emb)
            scored.append(
                ScoredDoc(
                    id=str(doc["_id"]),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=score,
                )
            )

        return sorted(scored, key=lambda d: d.score, reverse=True)[:top_k]


    # Helper: Text Search
    def _keyword_search(
        self,
        keyword_query: str,
        top_k: int,
        base_filter: Dict,
    ) -> List[ScoredDoc]:

        if not keyword_query:
            return []

        try:
            results = self.collection.aggregate([
                {"$search": {
                    "index": self.text_index,
                    "text": {
                        "query": keyword_query,
                        "path": self.text_paths,
                    }
                }},
                {"$limit": top_k},
                {"$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"},
                }},
            ])
        except Exception:
            logger.warning("Text index unavailable; falling back to $text")
            query = dict(base_filter)
            query["$text"] = {"$search": keyword_query}
            results = self.collection.find(
                query,
                {"content": 1, "metadata": 1, "score": {"$meta": "textScore"}},
            ).limit(top_k)

        return [
            ScoredDoc(
                id=str(d["_id"]),
                content=d.get("content", ""),
                metadata=d.get("metadata", {}),
                score=d.get("score", 0.0),
            )
            for d in results
        ]

    # Reciprocal Rank Fusion
    @staticmethod
    def _rrf_fuse(
        runs: List[List[ScoredDoc]],
        k: int,
        limit: int,
        weights: Optional[List[float]] = None,
    ) -> List[ScoredDoc]:

        scores = defaultdict(float)
        docs: Dict[str, ScoredDoc] = {}

        for i, run in enumerate(runs):
            weight = weights[i] if weights else 1.0
            for rank, doc in enumerate(run):
                docs[doc.id] = doc
                scores[doc.id] += weight / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [docs[doc_id] for doc_id, _ in ranked[:limit]]

# --- RAG Class ---

class RAG:
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/?directConnection=true",
        database: str = "smartlp",
        collection_name: str = "knowledge_base",
        embedding_dim: int = 384,
        embedding_provider: str = "all-MiniLM-L6-v2",
        vector_index: str = "vector_index",
        text_index: str = "text_index",
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
        self.collection: Optional[Collection] = None
        self.client: Optional[MongoClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self.embedding_fn = lambda texts, show_progress=False: self.generate_embeddings(texts, show_progress)
        self.local_retriever = LocalRetriever(
            collection=self._ensure_collection(),
            embedding_fn=self.embedding_fn,
            text_index=self.text_index,
        )


    # --- Index Creation Helpers ---
    def _ensure_index(
        self,
        collection: Collection,
        index_name: str,
        index_type: str,
        definition: dict
    ) -> None:
        """
        Ensure a search or vector index exists in the collection.
        """
        try:
            existing = list(collection.list_search_indexes(index_name))
            if existing:
                logger.info("%s index '%s' already exists.", index_type.capitalize(), index_name)
                return
        except Exception as exc:
            logger.debug("Index existence check failed: %s", exc)

        try:
            collection.create_search_index(model={"definition": definition, "name": index_name, "type": index_type})
            logger.info("%s index '%s' creation initiated.", index_type.capitalize(), index_name)
        except OperationFailure as e:
            if "already exists" in str(e):
                logger.info("%s index '%s' already exists.", index_type.capitalize(), index_name)
            else:
                logger.error("Failed to create %s index '%s': %s", index_type, index_name, e)

    # --- Connection / index helpers ---
    def connect(self) -> MongoClient:
        if self.client is None:
            try:
                client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                self.client = client
            except ServerSelectionTimeoutError as exc:
                logger.error("Unable to reach MongoDB at %s: %s", self.mongo_uri, exc)
                raise
        return self.client

    def _ensure_collection(self) -> Collection:
        if self.collection is None:
            client = self.connect()
            self.collection = client[self.database][self.collection_name]
        return self.collection

    def init(self) -> None:
        coll = self._ensure_collection()

        # --- Vector index definition ---
        vector_def = {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": self.embedding_dim, "similarity": "cosine"},
                {"type": "filter", "path": "metadata.category"}
            ]
        }

        # --- Text index definition ---
        text_def = {"mappings": {"dynamic": True}}

        # --- Ensure both indexes ---
        self._ensure_index(coll, self.vector_index, "vectorSearch", vector_def)
        self._ensure_index(coll, self.text_index, "search", text_def)

        logger.info("RAG initialization complete")

    # --- Embeddings ---
    def get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info("Loading SentenceTransformer: %s", self.embedding_provider)
            self._embedding_model = SentenceTransformer(self.embedding_provider)
        return self._embedding_model

    def generate_embeddings(self, texts: Sequence[str], show_progress: bool = False) -> List[List[float]]:
        embedder = self.get_embedding_model()
        embeddings = embedder.encode(texts, show_progress_bar=show_progress)
        return np.asarray(embeddings).tolist()


    # --- Document loading / chunking ---
    def load_documents(self, input_path: Path) -> List[Document]:
        from langchain_community.document_loaders import JSONLoader, PyPDFLoader, TextLoader

        def load_file(path: Path) -> List[Document]:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                return []
            
            try:
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
            except Exception as e:
                logger.error("Failed to load %s: %s", path, e)
                return []

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
        
        content = chunk.page_content.strip()
        content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()

        return {
            "_id": content_hash,
            "chunk_id": content_hash,
            "content": content,
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
            logger.info("No documents found under %s", input_path)
            return 0
        chunks = self.chunk_documents(docs)
        if not chunks:
            logger.info("No chunks produced")
            return 0
        logger.info("Loaded %s docs -> %s chunks", len(docs), len(chunks))

        # Delta check
        content_hashes = [hashlib.sha1(c.page_content.encode("utf-8")).hexdigest() for c in chunks]
        existing_ids: set[str] = set()
        
        # Check existence in batches
        check_batch_size = 5000
        for i in range(0, len(content_hashes), check_batch_size):
            batch = content_hashes[i: i + check_batch_size]
            if not batch: continue
            for doc in coll.find({"_id": {"$in": batch}}, {"_id": 1}):
                existing_ids.add(doc["_id"])
        
        chunks_to_embed = [chunk for chunk, h in zip(chunks, content_hashes) if h not in existing_ids]
        if not chunks_to_embed:
            logger.info("All chunks already exist; skipping ingest")
            return 0

        mongo_batch: List[Dict] = []
        inserted = 0

        for chunk_batch in tqdm(batched(chunks_to_embed, self.embedding_batch_size)):
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
                            inserted += exc.details.get("nInserted", 0)
                    else:
                        inserted += len(mongo_batch)
                    mongo_batch.clear()

        if mongo_batch and not dry_run:
            try:
                res = coll.insert_many(mongo_batch, ordered=False)
                inserted += len(res.inserted_ids)
            except BulkWriteError as exc:
                inserted += exc.details.get("nInserted", 0)
        elif mongo_batch and dry_run:
            inserted += len(mongo_batch)

        logger.info("Ingest complete. %s chunks %s", inserted, "simulated" if dry_run else "inserted")
        return inserted


    # --- Chain builder ---
    def _build_chain(self, retriever: MongoHybridRetriever, model_override=None, url_override=None, api_key_override=None) -> RunnableLambda:
        
        llm = ChatOpenAI(
            model=model_override or "qwen25-coder-32b-awq",
            base_url=url_override or "https://192.168.125.31:8000/v1",
            api_key=api_key_override or "testing",
            temperature=0
        )

        prompt = PromptTemplate(
            template="{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nUsing only the context above, provide the answer.",
            input_variables=["system_prompt", "question", "context"],
        )

        return (
            {
                "system_prompt": lambda x: x["system_prompt"],
                "question": lambda x: x["question"],
                "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    # --- Query Method ---
    def query_rag(self, user_prompt: str, system_prompt: Optional[str] = None, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        start = time.time()
        try:
            coll = self._ensure_collection()
            retriever = MongoHybridRetriever(
                collection=coll,
                embedding_fn=self.embedding_fn,
                embedding_dim=self.embedding_dim,
                vector_index=self.vector_index,
                text_index=self.text_index,
                top_k=top_k,
                semantic_candidates=kwargs.get("semantic_candidates", 50),
                keyword_candidates=kwargs.get("keyword_candidates", 30),
                rrf_k=kwargs.get("rrf_k", 60),
                allowed_text_paths=kwargs.get("allowed_text_paths", DEFAULT_TEXT_PATHS),
                fallback_engine=self.local_retriever,
                filter_category=kwargs.get("filter_category"),
            )
            
            chain = self._build_chain(retriever, kwargs.get("model_override"), kwargs.get("url_override"), kwargs.get("api_key_override"))
            answer = chain.invoke({"system_prompt": system_prompt or "", "question": user_prompt})

            return {"success": True, "content": answer, "latency": round(time.time() - start, 3)}

        except Exception as e:
            return {"success": False, "error": str(e), "latency": round(time.time() - start, 3)}

rag_service = RAG()

# --- CLI ---
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["init", "ingest", "query"])
    parser.add_argument("--query-text", default="Test query")
    parser.add_argument("--input-path", type=Path)
    args = parser.parse_args()

    rag = RAG() # Uses new defaults

    if args.mode == "init":
        rag.init()
    elif args.mode == "ingest" and args.input_path:
        rag.ingest(args.input_path)
    elif args.mode == "query":
        logger.info("%s", rag.query_rag(args.query_text))

if __name__ == "__main__":
    main()