"""Configuration settings for the RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the current directory
load_dotenv(Path(__file__).parent / ".env")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# Paths
ELASTIC_INTEGRATIONS_PATH = Path(os.getenv("ELASTIC_INTEGRATIONS_PATH", r"C:\Users\geola\Documents\GitHub\elastic_integrations\packages"))
DATA_DIR = Path("data")
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "elastic_integrations")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(DATA_DIR / "chroma_db"))

# Vector Database Settings
VECTOR_DIMENSION = 768  # Typical dimension for sentence transformers
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)
