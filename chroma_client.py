"""ChromaDB client functions for vector database operations."""

import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
from config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, TOP_K_RESULTS

logger = logging.getLogger(__name__)

# Global client instance
_chroma_client = None
_collection = None

def connect_to_chroma() -> bool:
    """
    Connect to ChromaDB database.
    
    Returns:
        True if connection successful, False otherwise
    """
    global _chroma_client
    
    try:
        # Try to connect to ChromaDB server first, fallback to persistent local storage
        try:
            _chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT
            )
            # Test the connection
            _chroma_client.heartbeat()
            logger.info(f"Connected to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception:
            # Fallback to persistent local storage
            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Connected to local ChromaDB at {CHROMA_PERSIST_DIRECTORY}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return False

def create_collection_if_not_exists() -> bool:
    """
    Create the collection for storing integration embeddings if it doesn't exist.
    
    Returns:
        True if collection exists or was created successfully
    """
    global _collection
    
    try:
        if not _chroma_client:
            if not connect_to_chroma():
                return False
        
        # Try to get existing collection
        try:
            _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")
            return True
        except Exception:
            # Collection doesn't exist, create it
            _collection = _chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Elastic integrations embeddings"}
            )
            logger.info(f"Created collection '{COLLECTION_NAME}'")
            return True
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False

def insert_integration_data(integrations_data: List[Dict[str, Any]]) -> bool:
    """
    Insert integration data with embeddings into ChromaDB.
    
    Args:
        integrations_data: List of integration dictionaries with embeddings
        
    Returns:
        True if insertion successful
    """
    try:
        if not _collection:
            if not create_collection_if_not_exists():
                return False
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, item in enumerate(integrations_data):
            ids.append(f"integration_{i}_{item['integration_name']}")
            embeddings.append(item["embedding"])
            
            # Prepare metadata (ChromaDB doesn't support nested objects directly)
            metadata = {
                "integration_name": item["integration_name"],
                "log_type": item["log_type"],
                "package_path": item["package_path"]
            }
            
            # Add additional metadata fields if they exist
            if "metadata" in item and isinstance(item["metadata"], dict):
                for key, value in item["metadata"].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"meta_{key}"] = value
                    elif isinstance(value, list) and value:
                        # Convert lists to comma-separated strings
                        if all(isinstance(x, str) for x in value):
                            metadata[f"meta_{key}"] = ", ".join(value)
            
            metadatas.append(metadata)
            documents.append(item["description"])
        
        # Insert data into ChromaDB
        _collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Inserted {len(integrations_data)} integration records")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")
        return False

def search_similar_integrations(query_embedding: List[float], top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """
    Search for similar integrations using vector similarity.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of top results to return
        
    Returns:
        List of similar integration records with scores
    """
    try:
        if not _collection:
            if not create_collection_if_not_exists():
                return []
        
        # Perform similarity search
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        similar_integrations = []
        
        if results['ids'] and results['ids'][0]:  # Check if we have results
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity_score = 1.0 - distance
                
                similar_integrations.append({
                    "integration_name": metadata.get("integration_name", ""),
                    "log_type": metadata.get("log_type", ""),
                    "description": document,
                    "package_path": metadata.get("package_path", ""),
                    "similarity_score": similarity_score,
                    "id": results['ids'][0][i]
                })
                
        return similar_integrations
        
    except Exception as e:
        logger.error(f"Failed to search similar integrations: {e}")
        return []

def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the collection.
    
    Returns:
        Dictionary with collection statistics
    """
    try:
        if not _collection:
            if not create_collection_if_not_exists():
                return {
                    "total_entities": 0,
                    "collection_name": COLLECTION_NAME,
                    "is_loaded": False,
                    "error": "Failed to connect to collection"
                }
        
        # Get collection count
        count_result = _collection.count()
        
        return {
            "total_entities": count_result,
            "collection_name": COLLECTION_NAME,
            "is_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {
            "total_entities": 0,
            "collection_name": COLLECTION_NAME,
            "is_loaded": False,
            "error": str(e)
        }

def clear_collection() -> bool:
    """
    Clear all data from the collection.
    
    Returns:
        True if successful
    """
    global _collection
    
    try:
        if not _chroma_client:
            if not connect_to_chroma():
                return False
        
        # Delete existing collection if it exists
        try:
            _chroma_client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted collection '{COLLECTION_NAME}'")
        except Exception:
            # Collection might not exist, which is fine
            pass
        
        # Create new empty collection
        return create_collection_if_not_exists()
        
    except Exception as e:
        logger.error(f"Failed to clear collection: {e}")
        return False

def disconnect_from_chroma():
    """Disconnect from ChromaDB database."""
    global _chroma_client, _collection
    
    try:
        # ChromaDB client doesn't need explicit disconnection
        # Just reset the global variables
        _chroma_client = None
        _collection = None
        logger.info("Disconnected from ChromaDB")
    except Exception as e:
        logger.warning(f"Error disconnecting from ChromaDB: {e}")

# Alias functions to maintain compatibility with existing code
connect_to_milvus = connect_to_chroma
disconnect_from_milvus = disconnect_from_chroma
