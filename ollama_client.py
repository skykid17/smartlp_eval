"""Ollama client functions for LLM and embedding operations."""

import json
import logging
from typing import List, Dict, Any, Optional
import requests
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for text in texts:
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            embedding = response.json().get("embedding", [])
            if embedding:
                embeddings.append(embedding)
            else:
                logger.warning(f"No embedding returned for text: {text[:50]}...")
                embeddings.append([0.0] * 768)  # Default dimension
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate embedding: {e}")
            embeddings.append([0.0] * 768)  # Default embedding
            
    return embeddings

def analyze_log_type(log_content: str) -> Dict[str, Any]:
    """
    Analyze log content to determine its type using LLM.
    
    Args:
        log_content: Raw log content as string
        
    Returns:
        Dictionary containing log type analysis
    """
    prompt = f"""Determine the specific log type from a given log entry. 
Analyze the log format, including timestamp style, field structure, severity labels, or known keywords.
Use domain-specific clues (e.g., paths, methods, IPs for web logs; query/error keywords for databases).
Match the log entry to one of the predefined types based on these patterns.
Return the exact log type identifier (e.g., "nginx_access").
If the log does not clearly match any type or you're unsure, return "unknown".

Return only the matched log_type string, e.g., "kubernetes_log". No explanations, no extra text.
Log entry to analyze:
{log_content[:1000]}"""
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 20
                }
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "unknown").strip()
        
        # Clean up the response - remove any extra formatting and normalize
        log_type = response_text.replace('"', '').replace("'", '').strip().lower()
        
        # Remove any extra text that might come after the log type
        log_type = log_type.split('\n')[0].split('.')[0].split(',')[0].strip()
        
        # Replace spaces with underscores for consistency
        log_type = log_type.replace(' ', '_').replace('-', '_')
        
        # Keep the specific log type returned by LLM instead of mapping to generic types
        if not log_type or log_type == 'unknown':
            log_type = 'application'  # fallback
            
        # Extract basic characteristics from the log type and content
        characteristics = []
        log_lower = log_content.lower()
        
        # Add characteristics based on specific log type
        if 'apache' in log_type:
            characteristics.extend(['apache', 'web_server'])
        elif 'nginx' in log_type:
            characteristics.extend(['nginx', 'web_server'])
        elif 'iis' in log_type:
            characteristics.extend(['iis', 'web_server'])
        elif any(db in log_type for db in ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']):
            characteristics.extend(['database'])
        elif 'syslog' in log_type or 'windows' in log_type:
            characteristics.extend(['system'])
        elif any(sec in log_type for sec in ['fortinet', 'cisco', 'palo_alto', 'checkpoint']):
            characteristics.extend(['security', 'firewall'])
        elif any(app in log_type for app in ['kafka', 'docker', 'kubernetes', 'jenkins']):
            characteristics.extend(['application'])
        
        # Add general characteristics based on content
        if 'access' in log_type or any(word in log_lower for word in ['get', 'post', 'http']):
            characteristics.append('access_log')
        if 'error' in log_type or '[error]' in log_lower or 'failed' in log_lower:
            characteristics.append('error_log')
        if any(word in log_lower for word in ['auth', 'login', 'security']):
            characteristics.append('security')
        if any(word in log_lower for word in ['http', 'get', 'post', 'status']):
            characteristics.append('http_request')
            
        # Detect service name from log content and log type
        service_name = None
        
        # Extract service name from log_type if it contains it
        if '_' in log_type:
            potential_service = log_type.split('_')[0]
            if potential_service in ['apache', 'nginx', 'iis', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'fortinet', 'cisco', 'kafka', 'docker', 'kubernetes']:
                service_name = potential_service
        
        # Fallback to content-based detection
        if not service_name:
            services = {
                'apache': ['apache', 'httpd'],
                'nginx': ['nginx'],
                'iis': ['iis', 'microsoft'],
                'mysql': ['mysql'],
                'postgresql': ['postgres', 'postgresql'],
                'mongodb': ['mongo', 'mongodb'],
                'redis': ['redis'],
                'elasticsearch': ['elasticsearch', 'elastic'],
                'fortinet': ['fortinet', 'fortigate'],
                'cisco': ['cisco'],
                'kafka': ['kafka'],
                'docker': ['docker'],
                'kubernetes': ['kubernetes', 'k8s']
            }
            
            for service, keywords in services.items():
                if any(keyword in log_lower for keyword in keywords):
                    service_name = service
                    break
        
        return {
            "log_type": log_type,
            "confidence": 0.9,
            "characteristics": list(set(characteristics)),  # Remove duplicates
            "service_name": service_name
        }
        
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to analyze log type: {e}")
        return {
            "log_type": "unknown",
            "confidence": 0.0,
            "characteristics": [],
            "service_name": None
        }

def check_ollama_connection() -> bool:
    """
    Check if Ollama service is running and accessible.
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False

def ensure_models_available() -> Dict[str, bool]:
    """
    Check if required models are available in Ollama.
    
    Returns:
        Dictionary with model availability status
    """
    status = {
        "embedding_model": False,
        "llm_model": False
    }
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        available_models = [model["name"] for model in models_data.get("models", [])]
        
        status["embedding_model"] = any(EMBEDDING_MODEL in model for model in available_models)
        status["llm_model"] = any(LLM_MODEL in model for model in available_models)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to check model availability: {e}")
        
    return status
