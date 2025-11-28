Splunk Fields can be found on their CIM field reference documentation site.
https://docs.splunk.com/Documentation/CIM/6.1.0/User/Overview

Elastic Fields can be found on their ECS field reference documentation site.
https://www.elastic.co/docs/reference/ecs/ecs-field-reference

Elastic Logtypes can be found on their github integrations repository.
https://github.com/elastic/integrations/tree/main/packages

Splunk Sourcetypes must be extracted from the stanzas in props.conf within each Add On package folder.
The Add-ons must be downloaded from Splunkbase.
https://splunkbase.splunk.com/apps?page=1&keyword=add-on&filters=built_by%3Asplunk%2Fproduct%3Asplunk

Regex Generation Prompt
```
You are an expert in log parsing and regular expressions. Given a log entry, generate a pcre2 compatible 
regex pattern with named capture groups. Capture as many fields as possible. Do not capture multiple fields within a capture 
group. Do not use a 'catchall' capture group. Always use .*? within capture groups. Take into account field values with 
whitespaces. Replace all whitespaces outside of capture groups with the \s+ token. Escape any literal special characters 
and forward slashes within the regex. Return only the regex pattern. 
```

#### Switch LLMs
1. Identify process that is serving current llm.
```
nvidia-smi
```
2. Kill that process (stop VLLM)
```
sudo kill -9 <process_id>
```

#### Serve finetuned model on vllm
```
cd /home/rdpuser3/Downloads/
vllm serve /home/rdpuser3/Downloads/qwen-2.5-coder-finetuned --served-model-name qwen25-coder-32b-finetuned --chat-template /home/rdpuser3/Documents/soc_rag/alpaca_chat_template.jinja --host 0.0.0.0 --port 8001
```

#### Serve AWQ model (direct, rag, decomposed_rag) on vllm
```
cd /home/rdpuser3/Downloads/
vllm serve /home/rdpuser3/Downloads/Qwen2.5-Coder-32B-Instruct-AWQ --served-model-name qwen25-coder-32b-awq --host 0.0.0.0 --port 8000
```

#### Test Connection
```
curl http://192.168.125.31:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "qwen25-coder-32b-awq","prompt": "Hello world!"}'
```

---

## RAG Backends: Chroma and MongoDB

This repository includes two RAG backends that share similar goals but target different deployment scenarios:

- `rag_chroma.py`: local, file-based experimentation using ChromaDB.
- `rag_mongo.py`: MongoDB-backed RAG pipeline using Atlas Search / `mongot`.

### `rag_chroma.py` (Chroma)

- Builds sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2`) from files and folders.
- Uses a local Chroma collection (`persist_directory`) as the vector store.
- Provides `create_embeddings_from_path()` for ingestion with checkpointing and batching for large directories.
- Exposes `query_rag()` to run RetrievalQA over a chosen collection using a remote LLM.

### `rag_mongo.py` (MongoDB)

- CLI-driven workflow with modes:
	- `init`: create text and vector search indexes.
	- `ingest`: chunk and ingest a specific file or directory.
	- `ingest_all`: convenience mode that ingests common SOC datasets:
		- `splunk_fields` → `data/splunk/splunk_fields.csv`
		- `elastic_fields` → `data/elastic/elastic_fields.csv`
		- `splunk_packages` → `data/splunk/repo/`
		- `elastic_packages` → `data/elastic/repo/`
	- `query`: hybrid retrieval (vector + keyword) plus LLM answer generation.
	- `test`: quick vector-search health check.
- Uses the same `all-MiniLM-L6-v2` embeddings for consistency with the Chroma flow.
- Performs recursive text splitting and stores chunks plus curated metadata in MongoDB.
- Uses deterministic content-hash `_id`s and a pre-ingest delta check so re-running ingest only processes new/changed chunks.
- Supports category tagging and optional category filters at query time.
- Implements manual Reciprocal Rank Fusion (RRF) over vector and text search results for robust retrieval quality.