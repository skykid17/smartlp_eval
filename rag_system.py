"""Main RAG system for log analysis and integration recommendation."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from config import SIMILARITY_THRESHOLD, TOP_K_RESULTS
from log_preprocessor import preprocess_log
from ollama_client import analyze_log_type, generate_embeddings, check_ollama_connection
from chroma_client import (
    connect_to_chroma, search_similar_integrations, 
    get_collection_stats, disconnect_from_chroma
)
from integration_discovery import discover_integration_packages, create_integration_description

logger = logging.getLogger(__name__)

def initialize_system() -> Dict[str, bool]:
    """
    Initialize the RAG system by checking all dependencies.
    
    Returns:
        Dictionary with initialization status for each component
    """
    status = {
        "ollama_connection": False,
        "chroma_connection": False,
        "integration_data": False
    }
    
    try:
        # Check Ollama connection
        status["ollama_connection"] = check_ollama_connection()
        if not status["ollama_connection"]:
            logger.error("Failed to connect to Ollama service")
          # Check ChromaDB connection
        status["chroma_connection"] = connect_to_chroma()
        if not status["chroma_connection"]:
            logger.error("Failed to connect to ChromaDB database")
          # Check if integration data is available
        if status["chroma_connection"]:
            stats = get_collection_stats()
            status["integration_data"] = stats.get("total_entities", 0) > 0
            if not status["integration_data"]:
                logger.warning("No integration data found in ChromaDB collection")
        
        return status
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return status

def process_log_for_recommendation(log_content: str) -> Dict[str, Any]:
    """
    Process a log entry and generate recommendations.
    
    Args:
        log_content: Raw log content as string
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Preprocess the log
        logger.info("Preprocessing log content...")
        preprocessed_log = preprocess_log(log_content)
        
        if not preprocessed_log["cleaned_content"]:
            return {
                "status": "error",
                "message": "Log content is empty or invalid",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Step 2: Analyze log type using LLM
        logger.info("Analyzing log type...")
        log_analysis = analyze_log_type(preprocessed_log["normalized_content"])
        
        # Step 3: Generate embeddings for the processed log
        logger.info("Generating embeddings...")
        description_for_embedding = create_log_description_for_embedding(
            preprocessed_log, log_analysis
        )
        
        embeddings = generate_embeddings([description_for_embedding])
        if not embeddings or not embeddings[0]:
            return {
                "status": "error",
                "message": "Failed to generate embeddings",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Step 4: Search for similar integrations
        logger.info("Searching for similar integrations...")
        similar_integrations = search_similar_integrations(
            embeddings[0], top_k=TOP_K_RESULTS
        )
        
        # Step 5: Filter and rank recommendations
        recommendations = filter_and_rank_recommendations(
            similar_integrations, log_analysis, preprocessed_log
        )
        
        # Step 6: Prepare final response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "log_analysis": log_analysis,
            "preprocessed_log": {
                "metadata": preprocessed_log["metadata"],
                "patterns": preprocessed_log["patterns"],
                "token_count": len(preprocessed_log["tokens"])
            },
            "recommendations": recommendations,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add summary recommendation
        if recommendations:
            result["primary_recommendation"] = recommendations[0]
            result["recommendation_confidence"] = recommendations[0]["final_score"]
        else:
            result["primary_recommendation"] = None
            result["recommendation_confidence"] = 0.0
            result["message"] = "No suitable integration package found"
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing log for recommendation: {e}")
        return {
            "status": "error",
            "message": str(e),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

def create_log_description_for_embedding(
    preprocessed_log: Dict[str, Any], 
    log_analysis: Dict[str, Any]
) -> str:
    """
    Create a description string optimized for embedding generation.
    
    Args:
        preprocessed_log: Preprocessed log data
        log_analysis: LLM analysis results
        
    Returns:
        Description string for embedding
    """
    description_parts = []
    
    # Add log type from analysis
    log_type = log_analysis.get("log_type", "unknown")
    description_parts.append(f"Log type: {log_type}")
    
    # Add service name if detected
    service_name = log_analysis.get("service_name")
    if service_name:
        description_parts.append(f"Service: {service_name}")
    
    # Add characteristics
    characteristics = log_analysis.get("characteristics", [])
    if characteristics:
        char_str = ', '.join(characteristics)
        description_parts.append(f"Characteristics: {char_str}")
    
    # Add detected patterns
    patterns = preprocessed_log.get("patterns", [])
    if patterns:
        pattern_names = [p["pattern_name"] for p in patterns]
        pattern_str = ', '.join(pattern_names)
        description_parts.append(f"Patterns: {pattern_str}")
    
    # Add log levels if detected
    log_levels = preprocessed_log.get("metadata", {}).get("log_levels", [])
    if log_levels:
        levels_str = ', '.join(set(log_levels))
        description_parts.append(f"Log levels: {levels_str}")
    
    # Add IP addresses (indicates network/security logs)
    if preprocessed_log.get("metadata", {}).get("ip_addresses"):
        description_parts.append("Contains IP addresses (network/security related)")
    
    # Add status codes (indicates web server logs)
    status_codes = preprocessed_log.get("metadata", {}).get("status_codes", [])
    if status_codes:
        description_parts.append("Contains HTTP status codes (web server)")
    
    # Add file paths (indicates system logs)
    if preprocessed_log.get("metadata", {}).get("file_paths"):
        description_parts.append("Contains file paths (system related)")
    
    return ' | '.join(description_parts)

def filter_and_rank_recommendations(
    similar_integrations: List[Dict[str, Any]],
    log_analysis: Dict[str, Any],
    preprocessed_log: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter and rank integration recommendations based on various factors.
    
    Args:
        similar_integrations: List of similar integrations from vector search
        log_analysis: LLM analysis results
        preprocessed_log: Preprocessed log data
        
    Returns:
        Filtered and ranked list of recommendations
    """
    if not similar_integrations:
        return []
    
    recommendations = []
    
    for integration in similar_integrations:
        # Start with similarity score from vector search
        base_score = integration.get("similarity_score", 0.0)
        
        # Skip if similarity is too low
        if base_score < SIMILARITY_THRESHOLD:
            continue
        
        # Calculate additional scoring factors
        bonus_score = 0.0
        scoring_factors = []
        
        # Log type matching bonus
        detected_log_type = log_analysis.get("log_type", "").lower()
        integration_log_type = integration.get("log_type", "").lower()
        
        if detected_log_type in integration_log_type or integration_log_type in detected_log_type:
            bonus_score += 0.2
            scoring_factors.append("log_type_match")
        
        # Service name matching bonus
        service_name = log_analysis.get("service_name", "").lower()
        integration_name = integration.get("integration_name", "").lower()
        
        if service_name and service_name in integration_name:
            bonus_score += 0.3
            scoring_factors.append("service_name_match")
        
        # Pattern matching bonus
        patterns = [p["pattern_name"] for p in preprocessed_log.get("patterns", [])]
        for pattern in patterns:
            if pattern.lower() in integration.get("description", "").lower():
                bonus_score += 0.1
                scoring_factors.append(f"pattern_match_{pattern}")
                break
        
        # Calculate final score
        final_score = min(1.0, base_score + bonus_score)
        
        recommendation = {
            "integration_name": integration.get("integration_name"),
            "log_type": integration.get("log_type"),
            "description": integration.get("description"),
            "package_path": integration.get("package_path"),
            "similarity_score": base_score,
            "bonus_score": bonus_score,
            "final_score": final_score,
            "scoring_factors": scoring_factors,
            "confidence_level": get_confidence_level(final_score)
        }
        
        recommendations.append(recommendation)
    
    # Sort by final score (descending)
    recommendations.sort(key=lambda x: x["final_score"], reverse=True)
    
    return recommendations

def get_confidence_level(score: float) -> str:
    """
    Convert numerical score to confidence level string.
    
    Args:
        score: Numerical confidence score (0-1)
        
    Returns:
        Confidence level string
    """
    if score >= 0.9:
        return "very_high"
    elif score >= 0.8:
        return "high"
    elif score >= 0.7:
        return "medium"
    elif score >= 0.6:
        return "low"
    else:
        return "very_low"

def analyze_log(log_content: str) -> str:
    """
    Main function to analyze a log and return integration recommendation.
    
    Args:
        log_content: Raw log content as string
        
    Returns:
        Integration package name or "no package found"
    """
    try:
        # Initialize system if not already done
        init_status = initialize_system()
        if not all(init_status.values()):
            logger.error("System initialization failed")
            return "system_error"
        
        # Process log for recommendation
        result = process_log_for_recommendation(log_content)
        
        if result["status"] != "success":
            logger.error(f"Log processing failed: {result.get('message')}")
            return "processing_error"
        
        # Return primary recommendation or "no package found"
        primary_rec = result.get("primary_recommendation")
        if primary_rec and primary_rec["confidence_level"] in ["high", "very_high"]:
            return primary_rec["integration_name"]
        elif primary_rec and primary_rec["confidence_level"] == "medium":
            # Return with confidence indicator
            return f"{primary_rec['integration_name']} (medium_confidence)"
        else:
            return "no package found"
            
    except Exception as e:
        logger.error(f"Error in analyze_log: {e}")
        return "analysis_error"
    finally:        # Clean up connections
        disconnect_from_chroma()

def get_detailed_analysis(log_content: str) -> Dict[str, Any]:
    """
    Get detailed analysis results including all recommendations and metadata.
    
    Args:
        log_content: Raw log content as string
        
    Returns:
        Complete analysis results dictionary
    """
    try:
        # Initialize system
        init_status = initialize_system()
        if not all(init_status.values()):
            return {
                "status": "error",
                "message": "System initialization failed",
                "init_status": init_status
            }
          # Process log
        result = process_log_for_recommendation(log_content)
        result["init_status"] = init_status
        
        return result
    except Exception as e:
        logger.error(f"Error in get_detailed_analysis: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        disconnect_from_chroma()
