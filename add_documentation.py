#!/usr/bin/env python3
"""
Splunk Add-ons Documentation Extractor

This script processes folders in the splunk_add_ons directory and extracts
documentation from the splunk_compiled.xlsx file to create documentation.txt
files in each corresponding app folder.
"""

import os
import pandas as pd
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

def main():
    """Main function to process Splunk add-ons and extract documentation"""
    
    # Configuration
    SPLUNK_ADDONS_DIR = r"C:\Users\geola\Documents\GitHub\splunk_add_ons"
    EXCEL_FILE = "splunk_compiled.xlsx"
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_FILE):
        logger.error(f"Excel file not found: {EXCEL_FILE}")
        return
    
    # Check if splunk add-ons directory exists
    splunk_dir = Path(SPLUNK_ADDONS_DIR)
    if not splunk_dir.exists():
        logger.error(f"Splunk add-ons directory not found: {SPLUNK_ADDONS_DIR}")
        return
    
    try:
        # Load the Excel file
        logger.info(f"Loading Excel file: {EXCEL_FILE}")
        df = pd.read_excel(EXCEL_FILE)
        
        # Check if required columns exist
        if 'App' not in df.columns:
            logger.error("Column 'App' not found in Excel file")
            return

        if 'Description' not in df.columns:
            logger.error("Column 'Description' not found in Excel file")
            return
        
        logger.info(f"Excel file loaded successfully with {len(df)} rows")
        
        # Process each folder in the splunk add-ons directory
        processed_count = 0
        created_count = 0
        skipped_count = 0
        
        for folder in splunk_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                logger.info(f"Processing folder: {folder_name}")
                
                # Search for matching row in Excel file
                matching_rows = df[df['App'] == folder_name]
                
                if not matching_rows.empty:
                    # Get the first matching row
                    row = matching_rows.iloc[0]
                    documentation = row['Description']
                    
                    # Check if documentation content exists and is not NaN
                    if pd.notna(documentation) and str(documentation).strip():
                        # Create documentation.txt file in the app folder
                        doc_file_path = folder / "documentation.txt"
                        
                        try:
                            with open(doc_file_path, 'w', encoding='utf-8') as f:
                                f.write(str(documentation))
                            
                            logger.info(f"Created documentation.txt for {folder_name}")
                            created_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error writing documentation file for {folder_name}: {e}")
                    else:
                        logger.warning(f"No documentation content found for {folder_name}")
                        skipped_count += 1
                else:
                    logger.warning(f"No matching row found in Excel for folder: {folder_name}")
                    skipped_count += 1
                
                processed_count += 1
        
        # Summary
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total folders processed: {processed_count}")
        logger.info(f"Documentation files created: {created_count}")
        logger.info(f"Folders skipped (no match or empty docs): {skipped_count}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")

if __name__ == "__main__":
    main()