# SOC RAG - Log Analysis & Integration Recommendation System

A Retrieval-Augmented Generation (RAG) system that analyzes log entries and recommends appropriate Elastic integration packages based on log type and characteristics.

## 🎯 Overview

This system combines Large Language Models (LLM) with vector similarity search to:
- Analyze incoming log entries to determine their type and characteristics
- Generate embeddings for log content using Ollama's embedding models
- Search a vector database of Elastic integrations for the best match
- Recommend the most suitable integration package or return "no package found"

## 🏗️ Architecture

- **Python**: Core implementation with functional programming approach
- **Ollama**: Hosts LLM (Llama3.2) and embedding model (nomic-embed-text)
- **ChromaDB**: Vector database for storing and searching integration embeddings
- **Elastic Integrations**: Local repository of integration packages for analysis

## 🚀 Features

- **Log Preprocessing**: Cleans, normalizes, and extracts metadata from logs
- **Intelligent Analysis**: Uses LLM to identify log types and characteristics
- **Vector Search**: Finds similar integrations using cosine similarity
- **Smart Scoring**: Combines similarity scores with additional matching factors
- **CLI Interface**: Easy-to-use command-line tools
- **Detailed Analytics**: Comprehensive analysis results with confidence scores

## 📋 Prerequisites

### Required Services
1. **Ollama** - Running locally on port 11434
   ```powershell
   # Install Ollama and pull required models
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **ChromaDB** - Vector database for local storage and search
   ```powershell
   # ChromaDB runs locally or as HTTP server
   # Default: Local persistent storage (no additional setup required)
   # Optional: ChromaDB server mode on port 8000
   ```

3. **Elastic Integrations Repository** - Local copy at specified path
   ```powershell
   git clone https://github.com/elastic/integrations.git C:\Users\geola\Documents\GitHub\elastic_integrations
   ```

## 🛠️ Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd soc_rag
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env` (if not already present)
   - Update paths and settings as needed

## ⚙️ Configuration

Edit `.env` file to customize settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=elastic_integrations

# Paths
ELASTIC_INTEGRATIONS_PATH=C:\Users\geola\Documents\GitHub\elastic_integrations

# System Configuration
LOG_LEVEL=INFO
SIMILARITY_THRESHOLD=0.7
TOP_K_RESULTS=5
```

## 🎮 Usage

### Command Line Interface

**Validate System Setup**
```powershell
python cli.py validate
```

**Initialize Integration Data**
```powershell
# First time setup
python cli.py init-data

# Force refresh of data
python cli.py init-data --force
```

**Analyze Log Content**
```powershell
# Analyze a log file
python cli.py analyze -f path/to/logfile.log

# Get detailed analysis
python cli.py analyze -f path/to/logfile.log --detailed

# Analyze from stdin
echo "127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] \"GET /index.html HTTP/1.1\" 200 2326" | python cli.py analyze

# Save detailed results to file
python cli.py analyze -f logfile.log --detailed -o results.json
```

**System Statistics**
```powershell
python cli.py stats
```

**Test with Sample Logs**
```powershell
python cli.py test
```

### Programmatic Usage

```python
from rag_system import analyze_log, get_detailed_analysis

# Simple analysis
log_content = '''127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326'''
recommendation = analyze_log(log_content)
print(f"Recommended integration: {recommendation}")

# Detailed analysis
detailed_result = get_detailed_analysis(log_content)
print(f"Log type: {detailed_result['log_analysis']['log_type']}")
print(f"Confidence: {detailed_result['log_analysis']['confidence']}")
```

## 📊 System Workflow

1. **Log Ingestion**: Accept log content as input string
2. **Preprocessing**: Clean, normalize, and extract metadata
3. **LLM Analysis**: Identify log type and characteristics
4. **Embedding Generation**: Create vector representation of log
5. **Vector Search**: Find similar integrations in ChromaDB
6. **Smart Scoring**: Combine similarity with additional factors
7. **Recommendation**: Return best match or "no package found"

## 🧪 Example Outputs

### Simple Analysis
```
Recommended integration: apache
```

### Detailed Analysis
```json
{
  "status": "success",
  "log_analysis": {
    "log_type": "web_server",
    "confidence": 0.95,
    "characteristics": ["apache_format", "http_request", "status_code"],
    "service_name": "apache"
  },
  "recommendations": [
    {
      "integration_name": "apache",
      "log_type": "web, access, error",
      "final_score": 0.92,
      "confidence_level": "very_high",
      "scoring_factors": ["log_type_match", "service_name_match"]
    }
  ],
  "processing_time": 2.34
}
```

## 📁 Project Structure

```
soc_rag/
├── cli.py                    # Command-line interface
├── config.py                 # Configuration settings
├── rag_system.py            # Main RAG system logic
├── ollama_client.py         # Ollama API client
├── chroma_client.py         # ChromaDB client
├── log_preprocessor.py      # Log cleaning and processing
├── integration_discovery.py # Elastic integration discovery
├── data_initialization.py   # Vector database initialization
├── example_usage.py         # Usage examples
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
└── README.md               # This file
```

## 🔧 Troubleshooting

### Common Issues

**"Ollama service not available"**
- Ensure Ollama is running: `ollama serve`
- Check if models are pulled: `ollama list`
- Verify connection: `curl http://localhost:11434/api/tags`

**"ChromaDB connection failed"**
- Check ChromaDB configuration in .env file
- For HTTP mode: Verify ChromaDB server is running on configured port
- For local mode: Check write permissions to persist directory
- Test connection: `curl http://localhost:8000/api/v1/heartbeat` (HTTP mode only)

**"No integration data found"**
- Run data initialization: `python cli.py init-data`
- Check Elastic integrations path exists
- Verify read permissions on integration directory

**"Processing errors"**
- Check log file encoding (should be UTF-8)
- Ensure log content is not empty
- Review system logs: `tail -f rag_system.log`

### Performance Tips

- Use batch processing for multiple logs
- Adjust `TOP_K_RESULTS` for faster searches
- Lower `SIMILARITY_THRESHOLD` for more results
- Monitor memory usage with large log files

## 📈 Monitoring & Logging

The system logs to both console and `rag_system.log` file:
- **INFO**: Normal operations and results
- **WARNING**: Non-critical issues
- **ERROR**: Failed operations
- **DEBUG**: Detailed execution information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the functional programming style
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs
3. Validate system setup: `python cli.py validate`
4. Create an issue with detailed error information

