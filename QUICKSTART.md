# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **Git** for cloning repositories

> **Note**: This system uses ChromaDB for vector storage, which provides local persistence without requiring Docker or external services, making setup simpler than traditional vector databases.

### Step 1: Setup
Run the automated setup script:

**Windows PowerShell:**
```powershell
# Run the setup script
.\setup.ps1

# Or manually:
python setup.py
```

**Alternative Manual Setup:**
```powershell
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. ChromaDB will be used locally (no additional setup required)
# Optional: Start ChromaDB server if using HTTP mode
# chroma run --host localhost --port 8000

# 4. Start Ollama and pull models
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Clone Elastic integrations
git clone https://github.com/elastic/integrations.git C:\Users\geola\Documents\GitHub\elastic_integrations

# 6. Initialize system
python cli.py init-data
```

### Step 2: Validate Setup
```powershell
python cli.py validate
```

### Step 3: Test the System
```powershell
# Quick test
python cli.py test

# Run comprehensive tests
python test_system.py

# Try example usage
python example_usage.py
```

### Step 4: Analyze Your First Log
```powershell
# Analyze a log file
python cli.py analyze -f path\to\your\logfile.log

# Get detailed analysis
python cli.py analyze -f path\to\your\logfile.log --detailed

# Analyze from command line
echo "127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] \"GET /index.html HTTP/1.1\" 200 2326" | python cli.py analyze
```

## 🔧 Common Issues

### "Ollama service not available"
```powershell
# Start Ollama
ollama serve

# Check if models are available
ollama list

# Pull required models if missing
ollama pull llama3.2
ollama pull nomic-embed-text
```

### "ChromaDB connection failed"
```powershell
# Check ChromaDB configuration in .env file
# For local mode: Check write permissions to persist directory
mkdir -p ./data/chroma_db

# For HTTP mode: Start ChromaDB server
chroma run --host localhost --port 8000

# Test connection (HTTP mode only)
curl http://localhost:8000/api/v1/heartbeat
```

### "No integration data found"
```powershell
# Initialize or refresh data
python cli.py init-data --force
```

### "ChromaDB persistence issues"
```powershell
# Check permissions on data directory
icacls .\data\chroma_db

# Clear and reinitialize if corrupted
rmdir /s .\data\chroma_db
python cli.py init-data --force
```

## 📊 Example Outputs

### Simple Analysis
```
> echo "127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] \"GET /index.html HTTP/1.1\" 200 2326" | python cli.py analyze

Recommended integration: apache
```

### Detailed Analysis
```json
{
  "status": "success",
  "log_analysis": {
    "log_type": "web_server",
    "confidence": 0.95,
    "service_name": "apache"
  },
  "recommendations": [
    {
      "integration_name": "apache",
      "final_score": 0.92,
      "confidence_level": "very_high"
    }
  ],
  "processing_time": 2.34
}
```

## 🛠️ Development

### Project Structure
```
soc_rag/
├── cli.py                    # Command-line interface
├── rag_system.py            # Main RAG logic
├── ollama_client.py         # Ollama integration
├── chroma_client.py         # ChromaDB client
├── log_preprocessor.py      # Log processing
├── integration_discovery.py # Integration discovery
├── data_initialization.py   # Data setup
├── config.py               # Configuration
├── setup.py               # Setup script
├── test_system.py         # Test suite
└── example_usage.py       # Usage examples
```

### Adding New Log Types
1. Update log patterns in `log_preprocessor.py`
2. Add type mappings in `integration_discovery.py`
3. Test with new log samples

### Customizing Similarity Scoring
Edit scoring factors in `rag_system.py`:
- Adjust `SIMILARITY_THRESHOLD` in config
- Modify bonus scoring logic
- Add new matching criteria

### ChromaDB Configuration
Configure ChromaDB settings in `.env`:
- `CHROMA_PERSIST_DIRECTORY`: Local storage path
- `CHROMA_HOST`/`CHROMA_PORT`: For HTTP server mode
- `COLLECTION_NAME`: Vector collection name

## 📞 Support

1. **Check system status**: `python cli.py validate`
2. **View logs**: Check `rag_system.log`
3. **Run tests**: `python test_system.py`
4. **Get statistics**: `python cli.py stats`

For issues, check the main README.md troubleshooting section.
