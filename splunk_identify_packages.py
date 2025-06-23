#!/usr/bin/env python3
"""
Splunk Add-ons Props.conf Identifier

This script identifies folders in the splunk_add_ons directory that are missing
props.conf files in their "default" folder.
"""

import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_props_conf_existence():
    """Check for props.conf files in Splunk add-on default folders"""
    
    # Configuration
    SPLUNK_ADDONS_DIR = r"C:\Users\geola\Documents\GitHub\splunk_add_ons"
    
    # Check if splunk add-ons directory exists
    splunk_dir = Path(SPLUNK_ADDONS_DIR)
    if not splunk_dir.exists():
        logger.error(f"Splunk add-ons directory not found: {SPLUNK_ADDONS_DIR}")
        return
    
    logger.info(f"Checking folders in: {SPLUNK_ADDONS_DIR}")
    logger.info("Looking for props.conf files in 'default' subdirectories...")
    logger.info("=" * 60)
    
    # Lists to track results
    folders_with_props = []
    folders_without_props = []
    folders_without_default = []
    
    # Process each folder in the splunk add-ons directory
    for folder in sorted(splunk_dir.iterdir()):
        if folder.is_dir():
            folder_name = folder.name
            default_folder = folder / "default"
            props_conf_path = default_folder / "props.conf"
            
            if default_folder.exists():
                if props_conf_path.exists():
                    folders_with_props.append(folder_name)
                    logger.info(f"✓ {folder_name} - has props.conf")
                else:
                    folders_without_props.append(folder_name)
                    logger.warning(f"✗ {folder_name} - missing props.conf in default folder")
            else:
                folders_without_default.append(folder_name)
                logger.warning(f"✗ {folder_name} - no default folder exists")
    
    # Summary report
    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 60)
    
    logger.info(f"Total folders processed: {len(folders_with_props) + len(folders_without_props) + len(folders_without_default)}")
    logger.info(f"Folders with props.conf: {len(folders_with_props)}")
    logger.info(f"Folders missing props.conf: {len(folders_without_props)}")
    logger.info(f"Folders without default directory: {len(folders_without_default)}")
    
    # Detailed lists
    if folders_without_props:
        logger.info("\n" + "=" * 60)
        logger.info("FOLDERS MISSING props.conf IN DEFAULT DIRECTORY:")
        logger.info("=" * 60)
        for folder in folders_without_props:
            logger.info(f"  • {folder}")
    
    if folders_without_default:
        logger.info("\n" + "=" * 60)
        logger.info("FOLDERS WITHOUT DEFAULT DIRECTORY:")
        logger.info("=" * 60)
        for folder in folders_without_default:
            logger.info(f"  • {folder}")
    
    # Save results to file
    results_file = "splunk_props_conf_missing.txt"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Splunk Add-ons Props.conf Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total folders processed: {len(folders_with_props) + len(folders_without_props) + len(folders_without_default)}\n")
            f.write(f"Folders with props.conf: {len(folders_with_props)}\n")
            f.write(f"Folders missing props.conf: {len(folders_without_props)}\n")
            f.write(f"Folders without default directory: {len(folders_without_default)}\n\n")
            
            if folders_without_props:
                f.write("FOLDERS MISSING props.conf IN DEFAULT DIRECTORY:\n")
                f.write("-" * 50 + "\n")
                for folder in folders_without_props:
                    f.write(f"{folder}\n")
                f.write("\n")
            
            if folders_without_default:
                f.write("FOLDERS WITHOUT DEFAULT DIRECTORY:\n")
                f.write("-" * 50 + "\n")
                for folder in folders_without_default:
                    f.write(f"{folder}\n")
                f.write("\n")
            
            if folders_with_props:
                f.write("FOLDERS WITH props.conf (for reference):\n")
                f.write("-" * 50 + "\n")
                for folder in folders_with_props:
                    f.write(f"{folder}\n")
        
        logger.info(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")

def main():
    """Main function"""
    logger.info("Starting Splunk Add-ons Props.conf Analysis")
    check_props_conf_existence()
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()