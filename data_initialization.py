"""Data initialization functions to populate the vector database with integration data."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from config import DATA_DIR
from integration_discovery import discover_integration_packages, create_integration_description
from ollama_client import generate_embeddings, check_ollama_connection
from chroma_client import (
    connect_to_chroma, create_collection_if_not_exists, 
    insert_integration_data, clear_collection, get_collection_stats
)

logger = logging.getLogger(__name__)

def initialize_integration_data(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Initialize the vector database with integration data.
    
    Args:
        force_refresh: If True, clear existing data and rebuild
        
    Returns:
        Dictionary with initialization results
    """
    start_time = datetime.now()
    
    try:
        # Check system prerequisites
        if not check_ollama_connection():
            return {
                "status": "error",                "message": "Ollama service is not available",
                "processing_time": 0
            }
        
        if not connect_to_chroma():
            return {
                "status": "error", 
                "message": "Failed to connect to ChromaDB",
                "processing_time": 0
            }
        
        # Check if data already exists and force_refresh is False
        if not force_refresh:
            stats = get_collection_stats()
            if stats.get("total_entities", 0) > 0:
                return {
                    "status": "success",
                    "message": "Integration data already exists",
                    "total_integrations": stats["total_entities"],
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
        
        # Create or clear collection
        if force_refresh:
            logger.info("Clearing existing collection data...")
            if not clear_collection():
                return {
                    "status": "error",
                    "message": "Failed to clear collection",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
        else:
            if not create_collection_if_not_exists():
                return {
                    "status": "error",
                    "message": "Failed to create collection",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
        
        # Step 1: Discover integration packages
        logger.info("Discovering integration packages...")
        integrations = discover_integration_packages()
        
        if not integrations:
            return {
                "status": "warning",
                "message": "No integration packages found",
                "total_integrations": 0,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Step 2: Process integrations and generate embeddings
        logger.info(f"Processing {len(integrations)} integrations...")
        processed_integrations = []
        
        # Process in batches for better memory management
        batch_size = 10
        for i in range(0, len(integrations), batch_size):
            batch = integrations[i:i + batch_size]
            batch_results = process_integration_batch(batch)
            processed_integrations.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(integrations) + batch_size - 1)//batch_size}")
          # Step 3: Insert data into ChromaDB
        logger.info("Inserting data into ChromaDB...")
        if processed_integrations:
            success = insert_integration_data(processed_integrations)
            if not success:
                return {
                    "status": "error",
                    "message": "Failed to insert data into ChromaDB",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
        
        # Step 4: Save metadata for future reference
        save_initialization_metadata(processed_integrations)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "message": "Integration data initialized successfully",
            "total_integrations": len(processed_integrations),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize integration data: {e}")
        return {
            "status": "error",
            "message": str(e),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

def process_integration_batch(integrations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of integrations to generate embeddings.
    
    Args:
        integrations: List of integration metadata
        
    Returns:
        List of processed integrations with embeddings
    """
    processed = []
    
    # Create descriptions for embedding
    descriptions = []
    for integration in integrations:
        description = create_integration_description(integration)
        descriptions.append(description)
    
    # Generate embeddings for all descriptions in the batch
    try:
        embeddings = generate_embeddings(descriptions)
        
        # Combine integrations with their embeddings
        for i, integration in enumerate(integrations):
            if i < len(embeddings) and embeddings[i]:
                processed_integration = {
                    "integration_name": integration["integration_name"],
                    "log_type": ", ".join(integration.get("log_types", [])),
                    "description": create_integration_description(integration),
                    "package_path": integration["package_path"],
                    "embedding": embeddings[i],
                    "metadata": {
                        "title": integration.get("title", ""),
                        "version": integration.get("version", ""),
                        "categories": integration.get("categories", []),
                        "data_streams": integration.get("data_streams", [])
                    }
                }
                processed.append(processed_integration)
            else:
                logger.warning(f"Failed to generate embedding for {integration['integration_name']}")
                
    except Exception as e:
        logger.error(f"Failed to process integration batch: {e}")
        
    return processed

def save_initialization_metadata(processed_integrations: List[Dict[str, Any]]):
    """
    Save initialization metadata to file for future reference.
    
    Args:
        processed_integrations: List of processed integrations
    """
    try:
        metadata = {
            "initialization_timestamp": datetime.now().isoformat(),
            "total_integrations": len(processed_integrations),
            "integrations": [
                {
                    "integration_name": integration["integration_name"],
                    "log_type": integration["log_type"],
                    "package_path": integration["package_path"]
                }
                for integration in processed_integrations
            ]
        }
        
        metadata_file = DATA_DIR / "initialization_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved initialization metadata to {metadata_file}")
        
    except Exception as e:
        logger.error(f"Failed to save initialization metadata: {e}")

def load_initialization_metadata() -> Dict[str, Any]:
    """
    Load initialization metadata from file.
    
    Returns:
        Initialization metadata dictionary
    """
    try:
        metadata_file = DATA_DIR / "initialization_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load initialization metadata: {e}")
    
    return {}

def validate_system_setup() -> Dict[str, Any]:
    """
    Validate that the system is properly set up and ready to use.
    
    Returns:
        Validation results dictionary
    """
    results = {
        "ollama_available": False,
        "chroma_available": False,
        "collection_exists": False,
        "data_populated": False,
        "total_integrations": 0,
        "last_initialization": None
    }
    
    try:        # Check Ollama
        results["ollama_available"] = check_ollama_connection()
        
        # Check ChromaDB
        results["chroma_available"] = connect_to_chroma()
        if results["chroma_available"]:
            # Check collection and data
            stats = get_collection_stats()
            results["collection_exists"] = stats.get("is_loaded", False)
            results["total_integrations"] = stats.get("total_entities", 0)
            results["data_populated"] = results["total_integrations"] > 0
        
        # Check initialization metadata
        metadata = load_initialization_metadata()
        if metadata:
            results["last_initialization"] = metadata.get("initialization_timestamp")
          # Overall status
        results["system_ready"] = all([
            results["ollama_available"],
            results["chroma_available"],
            results["collection_exists"],
            results["data_populated"]
        ])
        
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        results["error"] = str(e)
    
    return results

def get_integration_statistics() -> Dict[str, Any]:
    """
    Get statistics about the integration data in the system.
    
    Returns:
        Statistics dictionary
    """
    try:
        stats = get_collection_stats()
        metadata = load_initialization_metadata()
        
        result = {
            "total_integrations": stats.get("total_entities", 0),
            "collection_status": "loaded" if stats.get("is_loaded") else "not_loaded",
            "last_update": metadata.get("initialization_timestamp"),
            "integrations_by_type": {}
        }
        
        # Analyze integration types from metadata
        if metadata.get("integrations"):
            type_counts = {}
            for integration in metadata["integrations"]:
                log_types = integration.get("log_type", "").split(", ")
                for log_type in log_types:
                    if log_type and log_type != "":
                        type_counts[log_type] = type_counts.get(log_type, 0) + 1
            
            result["integrations_by_type"] = type_counts
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get integration statistics: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize integration data
    result = initialize_integration_data(force_refresh=True)
    print(json.dumps(result, indent=2))
