import os
import json
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm import tqdm

import chromadb

# ---------- CONFIG ----------
VLLM_BASE_URL = "http://192.168.125.31:8000/v1"
VLLM_API_KEY = "test"
VLLM_MODEL_NAME = "qwen25-coder-32b-awq"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PATHS_FILE = "data/eval/input/all_document_paths.json"
OUTPUT_FILE = "data/eval/input/golden_dataset.json"

MIN_DOC_LENGTH = 50
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

CONTEXTS_PER_DOC = 3
TOP_K_RETRIEVAL = 5
CHUNK_QUALITY_THRESHOLD = 0.5   # new: threshold for chunk quality
MAX_CHUNK_IN_CONTEXT = 3

NUM_EVOLUTIONS = 1
EVOLUTION_STRATEGIES = ["REASONING", "MULTICONTEXT", "CONCRETIZE", "CONSTRAINED"]
GEN_TEMPERATURE = 0.2
GEN_MAX_TOKENS = 1024
CHECKPOINT_INTERVAL = 10

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("golden_generation_vllm.log"), logging.StreamHandler()],
)

class QA_Pair(BaseModel):
    input: str = Field(description="The question based on the context")
    expected_output: str = Field(description="The detailed answer to the question based on the context")

@dataclass
class Golden:
    input: str
    actual_output: Optional[str]
    expected_output: str
    context: List[str]
    source_file: str

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        if not isinstance(text, str):
            text = str(text)
        if text.startswith("content="):
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        return json.loads(text)
    except Exception:
        return {"input": text[:50], "expected_output": text}

def setup_vllm() -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL_NAME,
        openai_api_base=VLLM_BASE_URL,
        openai_api_key=VLLM_API_KEY,
        temperature=GEN_TEMPERATURE,
        max_tokens=GEN_MAX_TOKENS,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

def setup_embedder() -> SentenceTransformer:
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def chunk_document(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    return splitter.split_text(text)

def chunk_quality(chunk: str) -> float:
    # simple heuristic; you can replace with LLM-based critic
    words = chunk.strip().split()
    if len(words) < 20:
        return 0.0
    # e.g. prefer mid-size chunks
    return min(len(words) / 300.0, 1.0)

def index_all_documents(paths: List[str], embedder: SentenceTransformer, chroma_collection) -> None:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Failed to read {path}: {e}")
            continue
        if len(text.strip()) < MIN_DOC_LENGTH:
            continue
        chunks = chunk_document(text)
        # apply chunk-quality filter
        chunks = [c for c in chunks if chunk_quality(c) >= CHUNK_QUALITY_THRESHOLD]
        if not chunks:
            continue

        # Build canonical ids for the chunks (idempotent across runs)
        ids = [f"{os.path.basename(path)}::chunk_{i}" for i in range(len(chunks))]
        # Query Chroma for existing ids so we don't re-add duplicates
        try:
            existing = chroma_collection.get(ids=ids, include=["ids"])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            # If collection.get fails (older chroma SDKs or other), fall back to empty set
            existing_ids = set()

        # Determine indices which are missing in the index (need to be added)
        missing_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
        if not missing_indices:
            logging.info(f"No new chunks to add for {path} — all {len(ids)} chunks already indexed.")
            continue

        # Only embed the missing chunks (avoid re-embedding already indexed content)
        chunks_to_add = [chunks[i] for i in missing_indices]
        raw_embeddings = embedder.encode(chunks_to_add, convert_to_numpy=True)
        if isinstance(raw_embeddings, np.ndarray):
            embeddings = raw_embeddings.tolist()
        else:
            embeddings = [np.array(vec).tolist() for vec in raw_embeddings]

        ids_to_add = [ids[i] for i in missing_indices]
        metadatas = [{"source_file": path, "chunk_index": i} for i in missing_indices]
        docs_to_add = chunks_to_add
        try:
            chroma_collection.add(ids=ids_to_add, embeddings=embeddings, metadatas=metadatas, documents=docs_to_add)
            logging.info(f"Added {len(ids_to_add)} new chunks for {path} (skipped {len(ids) - len(ids_to_add)} existing chunks).")
        except Exception as e:
            logging.error(f"Failed to add chunks for {path}: {e}")
    # Important: persistence handled by the PersistentClient used in main

def retrieve_context_groups(chroma_collection, source_file: str, n_contexts: int) -> List[List[str]]:
    # fetch all chunks from that document via metadata filter
    res = chroma_collection.get(
        where={"source_file": source_file},
        include=["embeddings", "documents"]
    )
    all_chunks = res["documents"]
    all_ids = res["ids"]
    all_embs = res["embeddings"]
    groups = []
    for seed_idx in random.sample(range(len(all_ids)), min(n_contexts, len(all_ids))):
        seed_emb = all_embs[seed_idx]
        query = chroma_collection.query(query_embeddings=[seed_emb], n_results=TOP_K_RETRIEVAL)
        docs = query["documents"][0]
        groups.append(docs[:MAX_CHUNK_IN_CONTEXT])
    return groups

def evolve_question(llm: ChatOpenAI, question: str, context: str, strategy: str) -> str:
    templates = {
        "REASONING": "Rewrite the question to require multi-step reasoning. Output as a JSON object {\"input\": <rewritten question>}.",
        "MULTICONTEXT": "Rewrite the question to combine facts from multiple parts of the context. Output as a JSON object {\"input\": <rewritten question>}.",
        "CONCRETIZE": "Rewrite the question to make it more concrete and specific. Output as a JSON object {\"input\": <rewritten question>}.",
        "CONSTRAINED": "Rewrite the question to add constraints (e.g., list only, 3 bullets) while staying answerable. Output as a JSON object {\"input\": <rewritten question>}."
    }
    sys_prompt = (
        f"You are a question-improver. Original question: '{question}'.\n"
        f"Context:\n{context}\n"
        f"Instruction: {templates.get(strategy, 'Make it concrete')}\n"
        "Output only the rewritten question in the JSON format: {\"input\": <rewritten question>}."
    )
    try:
        resp = llm.invoke(sys_prompt).content
        return str(resp).strip()
    except Exception as e:
        logging.warning(f"Evolution failed ({strategy}): {e}")
        return question

def generate_for_context(context_chunks: List[str], llm: ChatOpenAI, parser: PydanticOutputParser, source_file: str) -> Optional[Dict[str, Any]]:
    context_text = "\n\n".join(context_chunks)
    sys_prompt = (
        "You are an expert at generating high-quality QA pairs.\n"
        "Generate a JSON {\"input\": <question>, \"expected_output\": <answer>}, based ONLY on the context below.\n"
        f"Context:\n{context_text}\n"
        f"{parser.get_format_instructions()}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=sys_prompt)])
        qa_obj = safe_parse_json(resp.content)
        gen_input = qa_obj.get("input", "").strip()
        gen_output = qa_obj.get("expected_output", "").strip()
    except Exception as e:
        logging.warning(f"QA generation failed for {source_file}: {e}")
        return None
    if not gen_input or not gen_output:
        return None

    evolved = gen_input
    for _ in range(NUM_EVOLUTIONS):
        strategy = random.choice(EVOLUTION_STRATEGIES)
        evolved = evolve_question(llm, evolved, context_text, strategy)
        evolved = safe_parse_json(evolved).get("input", evolved)
    sys_prompt_ans = (
        f"Context:\n{context_text}\nQuestion:\n{evolved}\n"
        "Generate a detailed answer using only the context. Output only the answer as a json {\"expected_output\": <answer>}."
    )
    try:
        final_answer = llm.invoke([HumanMessage(content=sys_prompt_ans)]).content.strip()
        final_answer = safe_parse_json(final_answer).get("expected_output", "").strip()
        print("final_answer:", final_answer)
    except Exception:
        final_answer = gen_output

    # optional: verification step
    valid = True
    try:
        verifier = (
            f"Context:\n{context_text}\nQuestion: {evolved}\nAnswer: {final_answer}\n"
            "Respond with JSON {\"status\": \"VALID\"} or {\"status\": \"INVALID\"}."
        )
        v = llm.invoke([HumanMessage(content=verifier)])
        data = safe_parse_json(v.content)
        valid = data.get("status", "").upper() == "VALID"
    except Exception:
        valid = False

    if not valid:
        logging.warning("Verification failed — using answer anyway.")

    return asdict(Golden(
        input=evolved,
        actual_output=None,
        expected_output=final_answer,
        context=context_chunks,
        source_file=source_file
    ))

def main():
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        doc_paths = json.load(f)

    llm = setup_vllm()
    embedder = setup_embedder()
    parser = PydanticOutputParser(pydantic_object=QA_Pair)

    # use PersistentClient for disk persistence
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("golden_chunks")

    # 1. Build/extend index (only need to run once or when new docs are added)
    index_all_documents(doc_paths, embedder, collection)

    all_goldens = []
    # include tqdm for progress tracking
    for path in tqdm(doc_paths):
        groups = retrieve_context_groups(collection, path, CONTEXTS_PER_DOC)
        for ctx in groups:
            g = generate_for_context(ctx, llm, parser, source_file=path)
            if g:
                all_goldens.append(g)

    write_json(OUTPUT_FILE, all_goldens)
    logging.info(f"Generated {len(all_goldens)} goldens, saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
