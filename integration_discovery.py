"""Functions for discovering and processing Elastic integrations."""

import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import ELASTIC_INTEGRATIONS_PATH

logger = logging.getLogger(__name__)

def discover_integration_packages() -> List[Dict[str, Any]]:
    """
    Discover all integration packages in the Elastic integrations directory.
    
    Returns:
        List of integration package information
    """
    integrations = []
    
    if not ELASTIC_INTEGRATIONS_PATH.exists():
        logger.warning(f"Elastic integrations path does not exist: {ELASTIC_INTEGRATIONS_PATH}")
        return integrations
    
    # Look for integration directories
    for integration_dir in ELASTIC_INTEGRATIONS_PATH.iterdir():
        if integration_dir.is_dir() and not integration_dir.name.startswith('.'):
            integration_info = parse_integration_metadata(integration_dir)
            if integration_info:
                integrations.append(integration_info)
    
    logger.info(f"Discovered {len(integrations)} integration packages")
    return integrations

def parse_integration_metadata(integration_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse metadata from an integration package directory.
    
    Args:
        integration_path: Path to integration directory
        
    Returns:
        Integration metadata dictionary or None if parsing fails
    """
    try:
        integration_name = integration_path.name
        
        # Look for common metadata files
        manifest_file = integration_path / "manifest.yml"
        readme_file = integration_path / "README.md"
        docs_dir = integration_path / "docs"
        
        metadata = {
            "integration_name": integration_name,
            "package_path": str(integration_path),
            "description": "",
            "log_types": [],
            "categories": [],
            "version": "unknown",
            "title": integration_name.replace("_", " ").title(),
            "data_streams": []
        }
        
        # Parse manifest.yml if it exists
        if manifest_file.exists():
            manifest_data = parse_manifest_file(manifest_file)
            if manifest_data:
                metadata.update(manifest_data)
        
        # Extract information from README
        if readme_file.exists():
            readme_info = parse_readme_file(readme_file)
            if readme_info:
                metadata.update(readme_info)
        
        # Look for data stream configurations
        data_streams = discover_data_streams(integration_path)
        if data_streams:
            metadata["data_streams"] = data_streams
            
        # Infer log types from data streams and directory structure
        inferred_log_types = infer_log_types(integration_path, metadata)
        metadata["log_types"].extend(inferred_log_types)
        
        # Remove duplicates from log_types
        metadata["log_types"] = list(set(metadata["log_types"]))
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to parse integration metadata for {integration_path}: {e}")
        return None

def parse_manifest_file(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse manifest.yml file for integration metadata.
    
    Args:
        manifest_path: Path to manifest.yml file
        
    Returns:
        Parsed metadata dictionary
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = yaml.safe_load(f)
        
        metadata = {}
        
        if 'name' in manifest_data:
            metadata['integration_name'] = manifest_data['name']
            
        if 'title' in manifest_data:
            metadata['title'] = manifest_data['title']
            
        if 'description' in manifest_data:
            metadata['description'] = manifest_data['description']
            
        if 'version' in manifest_data:
            metadata['version'] = manifest_data['version']
            
        if 'categories' in manifest_data:
            metadata['categories'] = manifest_data['categories']
            
        # Extract log types from policy templates or data streams
        log_types = set()
        if 'policy_templates' in manifest_data:
            for template in manifest_data['policy_templates']:
                if 'data_streams' in template:
                    for ds in template['data_streams']:
                        if 'type' in ds:
                            log_types.add(ds['type'])
                            
        metadata['log_types'] = list(log_types)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to parse manifest file {manifest_path}: {e}")
        return None

def parse_readme_file(readme_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse README.md file for additional integration information.
    
    Args:
        readme_path: Path to README.md file
        
    Returns:
        Extracted information dictionary
    """
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        metadata = {}
        
        # If description is not already set, try to extract from README
        lines = readme_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and len(line.strip()) > 20:
                metadata['description'] = line.strip()
                break
        
        # Look for log type mentions
        log_type_keywords = [
            'apache', 'nginx', 'mysql', 'postgresql', 'mongodb', 'redis',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'syslog',
            'windows', 'linux', 'audit', 'security', 'firewall', 'proxy',
            'application', 'web', 'database', 'system', 'network'
        ]
        
        readme_lower = readme_content.lower()
        detected_types = []
        for keyword in log_type_keywords:
            if keyword in readme_lower:
                detected_types.append(keyword)
        
        if detected_types:
            metadata['log_types'] = detected_types
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to parse README file {readme_path}: {e}")
        return None

def discover_data_streams(integration_path: Path) -> List[Dict[str, Any]]:
    """
    Discover data streams within an integration package.
    
    Args:
        integration_path: Path to integration directory
        
    Returns:
        List of data stream configurations
    """
    data_streams = []
    
    # Look for data_stream directories
    data_stream_base = integration_path / "data_stream"
    if data_stream_base.exists():
        for ds_dir in data_stream_base.iterdir():
            if ds_dir.is_dir():
                ds_info = {
                    "name": ds_dir.name,
                    "path": str(ds_dir),
                    "type": "logs"  # Default assumption
                }
                
                # Look for manifest.yml in data stream
                ds_manifest = ds_dir / "manifest.yml"
                if ds_manifest.exists():
                    try:
                        with open(ds_manifest, 'r', encoding='utf-8') as f:
                            ds_manifest_data = yaml.safe_load(f)
                        
                        if 'type' in ds_manifest_data:
                            ds_info['type'] = ds_manifest_data['type']
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse data stream manifest {ds_manifest}: {e}")
                
                data_streams.append(ds_info)
    
    return data_streams

def infer_log_types(integration_path: Path, metadata: Dict[str, Any]) -> List[str]:
    """
    Infer log types from integration directory structure and metadata.
    
    Args:
        integration_path: Path to integration directory
        metadata: Existing metadata
        
    Returns:
        List of inferred log types
    """
    inferred_types = []
    
    # Infer from integration name
    integration_name = integration_path.name.lower()
    
    type_mappings = {
        'apache': ['web', 'access', 'error'],
        'nginx': ['web', 'access', 'error'],
        'mysql': ['database', 'audit', 'error'],
        'postgresql': ['database', 'audit'],
        'mongodb': ['database', 'audit'],
        'redis': ['database', 'audit'],
        'docker': ['container', 'system'],
        'kubernetes': ['container', 'orchestration'],
        'aws': ['cloud', 'audit'],
        'azure': ['cloud', 'audit'],
        'gcp': ['cloud', 'audit'],
        'windows': ['system', 'security', 'application'],
        'linux': ['system', 'audit'],
        'syslog': ['system', 'network'],
        'firewall': ['security', 'network'],
        'proxy': ['network', 'security'],
        'audit': ['security', 'compliance'],
        'security': ['security', 'audit']
    }
    
    for keyword, types in type_mappings.items():
        if keyword in integration_name:
            inferred_types.extend(types)
    
    # Infer from data streams
    for ds in metadata.get('data_streams', []):
        ds_name = ds.get('name', '').lower()
        ds_type = ds.get('type', 'logs')
        
        inferred_types.append(ds_type)
        
        if 'access' in ds_name:
            inferred_types.append('access')
        elif 'error' in ds_name:
            inferred_types.append('error')
        elif 'audit' in ds_name:
            inferred_types.append('audit')
        elif 'security' in ds_name:
            inferred_types.append('security')
    
    return list(set(inferred_types))

def create_integration_description(metadata: Dict[str, Any]) -> str:
    """
    Create a comprehensive description for embedding generation.
    
    Args:
        metadata: Integration metadata
        
    Returns:
        Formatted description string
    """
    description_parts = []
    
    # Add title and basic description
    title = metadata.get('title', metadata.get('integration_name', 'Unknown'))
    description_parts.append(f"Integration: {title}")
    
    if metadata.get('description'):
        description_parts.append(f"Description: {metadata['description']}")
    
    # Add log types
    if metadata.get('log_types'):
        log_types_str = ', '.join(metadata['log_types'])
        description_parts.append(f"Log types: {log_types_str}")
    
    # Add categories
    if metadata.get('categories'):
        categories_str = ', '.join(metadata['categories'])
        description_parts.append(f"Categories: {categories_str}")
    
    # Add data streams info
    if metadata.get('data_streams'):
        ds_names = [ds.get('name', '') for ds in metadata['data_streams']]
        ds_str = ', '.join(filter(None, ds_names))
        if ds_str:
            description_parts.append(f"Data streams: {ds_str}")
    
    return ' | '.join(description_parts)
