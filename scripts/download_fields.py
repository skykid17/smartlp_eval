from bs4 import BeautifulSoup
import requests
import pandas as pd
from pathlib import Path

# Point to project root (3 levels up: file -> rag -> src -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Keep downloaded assets inside the shared input directory
INPUT_DIR = BASE_DIR / "data" / "eval" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_splunk_fields():
    url = "https://docs.splunk.com/Documentation/CIM/6.1.0/User/Overview"

    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    })

    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        exit(1)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all ul elements with class "list-unstyled"
    data_model_sections = soup.find_all('ul', {'class': "list-unstyled"})

    # List to store all links
    all_links = []

    # Extract all hrefs
    for ul in data_model_sections:
        links = ul.find_all('a')
        for link in links:
            href = link.get('href', '')
            if href and href not in all_links and href.startswith('/Documentation/CIM/'):
                all_links.append(href)

    if not all_links:
        print("No links found. Exiting.")
        exit(1)

    def scrape_cim_data(url):
        """
        Scrape CIM data from the given URL.
        
        Args:
            url (str): The URL of the CIM data page.
            
        Returns:
            list: A list of dictionaries containing the scraped CIM data.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        try:
            response = requests.get(url, headers=headers)
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
                        "Dataset_name": cells[0].text.strip(),
                        "Field_name": cells[1].text.strip(),
                        "Data_type": cells[2].text.strip(),
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

    all_cim_data = []

    for link in sorted(all_links):
        cim_data = scrape_cim_data(f"https://docs.splunk.com{link}")
        print(f"Scraped {len(cim_data)} records from https://docs.splunk.com{link}")
        all_cim_data.extend(cim_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_cim_data)
    # Save the DataFrame to a CSV file
    destination = INPUT_DIR / "splunk_fields.csv"
    df.to_csv(destination, index=False)
    print(f"CIM data scraped and saved to {destination} ({len(all_cim_data)} records)")

def download_elastic_fields():
    """
    Download the fields CSV file from the Elastic GitHub repository.
    """
    headers = {
    "Accept": "application/vnd.github.v3.raw"  # Get the raw file content
    }  

    try:
        response = requests.get("https://raw.githubusercontent.com/elastic/ecs/master/generated/csv/fields.csv", headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        destination = INPUT_DIR / "elastic_fields.csv"
        with open(destination, "wb") as file:
            file.write(response.content)
        print(f"Downloaded elastic_fields.csv successfully to {destination}.")
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
    # Set the repo and file path

download_splunk_fields()

download_elastic_fields()