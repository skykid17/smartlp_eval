"""Log preprocessing functions for cleaning and normalizing log data."""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_log_content(log_content: str) -> str:
    """
    Clean and normalize log content.
    
    Args:
        log_content: Raw log content
        
    Returns:
        Cleaned log content
    """
    if not log_content:
        return ""
    
    # Basic cleaning steps
    cleaned = log_content.strip()
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove null characters
    cleaned = cleaned.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    cleaned = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    return cleaned

def extract_log_metadata(log_content: str) -> Dict[str, Any]:
    """
    Extract metadata from log content such as timestamps, IP addresses, etc.
    
    Args:
        log_content: Raw log content
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        "timestamps": [],
        "ip_addresses": [],
        "log_levels": [],
        "status_codes": [],
        "file_paths": [],
        "urls": []
    }
    
    # Extract timestamps (various formats)
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',  # ISO format
        r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',      # US format
        r'\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}:\d{2}',      # EU format
        r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',       # Syslog format
    ]
    
    for pattern in timestamp_patterns:
        matches = re.findall(pattern, log_content)
        metadata["timestamps"].extend(matches)
    
    # Extract IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    metadata["ip_addresses"] = re.findall(ip_pattern, log_content)
    
    # Extract log levels
    log_level_pattern = r'\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE|CRITICAL)\b'
    metadata["log_levels"] = re.findall(log_level_pattern, log_content, re.IGNORECASE)
    
    # Extract HTTP status codes
    status_code_pattern = r'\b[1-5]\d{2}\b'
    metadata["status_codes"] = re.findall(status_code_pattern, log_content)
    
    # Extract file paths
    file_path_pattern = r'[A-Za-z]:[\\\/][\w\s\\\/\.-]+|\/[\w\s\/\.-]+'
    metadata["file_paths"] = re.findall(file_path_pattern, log_content)
    
    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    metadata["urls"] = re.findall(url_pattern, log_content)
    
    # Remove duplicates
    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = list(set(metadata[key]))
    
    return metadata

def normalize_log_format(log_content: str) -> str:
    """
    Normalize log format for better analysis.
    
    Args:
        log_content: Raw log content
        
    Returns:
        Normalized log content
    """
    # Start with cleaned content
    normalized = clean_log_content(log_content)
    
    # Normalize timestamp formats to ISO format where possible
    timestamp_replacements = [
        (r'(\d{2})/(\d{2})/(\d{4})\s(\d{2}:\d{2}:\d{2})', r'\3-\1-\2 \4'),  # US to ISO
        (r'(\d{2})-(\d{2})-(\d{4})\s(\d{2}:\d{2}:\d{2})', r'\3-\2-\1 \4'),  # EU to ISO
    ]
    
    for pattern, replacement in timestamp_replacements:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Normalize log levels to uppercase
    log_levels = ['debug', 'info', 'warn', 'warning', 'error', 'fatal', 'trace', 'critical']
    for level in log_levels:
        pattern = r'\b' + re.escape(level) + r'\b'
        normalized = re.sub(pattern, level.upper(), normalized, flags=re.IGNORECASE)
    
    return normalized

def tokenize_log_content(log_content: str) -> List[str]:
    """
    Tokenize log content into meaningful tokens.
    
    Args:
        log_content: Log content to tokenize
        
    Returns:
        List of tokens
    """
    # Basic tokenization
    tokens = re.findall(r'\w+|[^\w\s]', log_content.lower())
    
    # Filter out very short tokens (less than 2 characters) and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    filtered_tokens = [token for token in tokens if len(token) >= 2 and token not in stop_words]
    
    return filtered_tokens

def identify_log_patterns(log_content: str) -> List[Dict[str, Any]]:
    """
    Identify common log patterns and structures.
    
    Args:
        log_content: Log content to analyze
        
    Returns:
        List of identified patterns
    """
    patterns = []
    
    # Common log format patterns
    format_patterns = {
        "apache_common": r'\d+\.\d+\.\d+\.\d+ - - \[.*?\] ".*?" \d+ \d+',
        "apache_combined": r'\d+\.\d+\.\d+\.\d+ - - \[.*?\] ".*?" \d+ \d+ ".*?" ".*?"',
        "nginx_access": r'\d+\.\d+\.\d+\.\d+ - \w+ \[.*?\] ".*?" \d+ \d+ ".*?" ".*?"',
        "syslog": r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+\w+.*?:',
        "windows_event": r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+\d+',
        "json_log": r'\{".*?"\s*:\s*".*?".*?\}',
    }
    
    for pattern_name, pattern_regex in format_patterns.items():
        matches = re.findall(pattern_regex, log_content)
        if matches:
            patterns.append({
                "pattern_name": pattern_name,
                "pattern_type": "format",
                "matches": len(matches),
                "sample": matches[0] if matches else None
            })
    
    return patterns

def preprocess_log(log_content: str) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for log content.
    
    Args:
        log_content: Raw log content
        
    Returns:
        Dictionary containing all preprocessing results
    """
    if not log_content or not log_content.strip():
        return {
            "original_content": log_content,
            "cleaned_content": "",
            "normalized_content": "",
            "metadata": {},
            "tokens": [],
            "patterns": [],
            "preprocessing_timestamp": datetime.now().isoformat()
        }
    
    # Apply preprocessing steps
    cleaned = clean_log_content(log_content)
    normalized = normalize_log_format(cleaned)
    metadata = extract_log_metadata(log_content)
    tokens = tokenize_log_content(normalized)
    patterns = identify_log_patterns(log_content)
    
    return {
        "original_content": log_content,
        "cleaned_content": cleaned,
        "normalized_content": normalized,
        "metadata": metadata,
        "tokens": tokens,
        "patterns": patterns,
        "preprocessing_timestamp": datetime.now().isoformat()
    }
