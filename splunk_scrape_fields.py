from bs4 import BeautifulSoup
import requests
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_cim_data(url):
    """
    Scrape CIM data from the given URL.
    
    Args:
        url (str): The URL of the CIM data page.
        
    Returns:
        list: A list of dictionaries containing the scraped CIM data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
          # Example of scraping logic; adjust according to actual HTML structure
        cim_data_list = []
        
        # Find all tables on the page (there might be multiple)
        tables = soup.find_all('table')
        
        for table_index, table in enumerate(tables):
            # Get headers to check the number of columns
            headers = [header.text.strip() for header in table.find_all('th')]
            
            # Skip tables that don't have exactly 5 columns
            if len(headers) != 5:
                print(f"Table {table_index + 1}: Found {len(headers)} columns, expected 5. Skipping table.")
                continue
                
            print(f"Table {table_index + 1}: Processing table with 5 columns: {headers}")
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) != 5:  # Only process rows with exactly 5 cells
                    continue
                
                row_data = {
                    "Dataset name": cells[0].text.strip(),
                    "Field name": cells[1].text.strip(),
                    "Data type": cells[2].text.strip(),
                    "Description": cells[3].text.strip(),
                    "Notes": cells[4].text.strip()  # Always save as "Notes" regardless of original column name
                }
                cim_data_list.append(row_data)
        
        return cim_data_list
    
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing data from {url}: {e}")
        return []

def main():
    data_models = [
        "Alerts",
        "Authentication",
        "Certificates",
        "Change",
        "DataAccess",
        "Datastore",
        "DataLossPrevention",
        "Email",
        "Endpoint",
        "EventSignatures",
        "InterprocessMessaging",
        "IntrusionDetection",
        "ComputeInventory",
        "Malware",
        "NetworkResolutionDNS",
        "NetworkSessions",
        "NetworkTraffic",
        "Performance",
        "SplunkAuditLogs",
        "TicketManagement",
        "Updates",
        "Vulnerabilities",
        "Web"
    ]
    
    all_cim_data = []
    data_extraction_log = {}  # Track data extraction per model
    
    for model in data_models:
        print(f"Scraping CIM data for model: {model}...")
        logging.info(f"Starting scraping for data model: {model}")
        
        cim_data = scrape_cim_data(f"https://docs.splunk.com/Documentation/CIM/6.1.0/User/{model}")
        data_count = len(cim_data)
        data_extraction_log[model] = data_count
        
        if data_count > 0:
            logging.info(f"Successfully extracted {data_count} records from {model}")
        else:
            logging.warning(f"No data extracted from {model} - possible issue with page structure or URL")
        
        all_cim_data.extend(cim_data)  # Use extend to add all rows from this model
    
    # Log summary of data extraction
    logging.info("=== DATA EXTRACTION SUMMARY ===")
    successful_models = []
    failed_models = []
    
    for model, count in data_extraction_log.items():
        if count > 0:
            successful_models.append(f"{model}: {count} records")
        else:
            failed_models.append(model)
    
    if successful_models:
        logging.info(f"Models with data extracted ({len(successful_models)}):")
        for model_info in successful_models:
            logging.info(f"  - {model_info}")
    
    if failed_models:
        logging.error(f"Models with NO data extracted ({len(failed_models)}):")
        for model in failed_models:
            logging.error(f"  - {model}")
        logging.error("Please check these models manually - they may have different page structures or URLs")
    else:
        logging.info("All data models successfully extracted data!")
    
    logging.info(f"Total records extracted: {len(all_cim_data)}")
    
    # Save the scraped data to a CSV file
    print("Saving CIM data to CSV...")
    if not all_cim_data:
        logging.error("No CIM data found from any model. Exiting.")
        print("No CIM data found. Exiting.")
        return

    df = pd.DataFrame(all_cim_data)
    df.to_csv('splunk_fields.csv', index=False)
    logging.info(f"CIM data saved to splunk_fields.csv ({len(all_cim_data)} records)")
    print(f"CIM data scraped and saved to splunk_fields.csv ({len(all_cim_data)} records)")
    print("Scraping complete.")

if __name__ == "__main__":
    main()