import os
import csv
import re
from pathlib import Path

# Point to project root (3 levels up: file -> generator -> src -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "eval" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_elastic_logtypes():
    # Return a list of sub-directories in the Elastic packages directory
    packages_dir = "repos/elastic_repo/packages"
    try:
        logtypes = [d for d in os.listdir(packages_dir) if os.path.isdir(os.path.join(packages_dir, d))]
        
        # Write logtypes to CSV file
        csv_filename = INPUT_DIR / "elastic_logtypes.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Package', 'Logtype', 'Documentation_URL'])
            
            # Write data rows
            for logtype in logtypes:
                documentation_url = f"https://github.com/elastic/integrations/tree/main/packages/{logtype}"
                writer.writerow([logtype, logtype, documentation_url])
        
        print(f"Successfully wrote {len(logtypes)} logtypes to '{csv_filename}'")
        return logtypes
        
    except FileNotFoundError:
        print(f"Directory '{packages_dir}' not found.")
        return []


def extract_splunk_sourcetypes():
    """Extract sourcetypes from all Splunk packages and write to CSV."""
    packages_dir = "repos/splunk_repo"
    csv_filename = INPUT_DIR / "splunk_sourcetypes.csv"
    
    try:
        # Get all subdirectories in splunk_repo
        package_names = [d for d in os.listdir(packages_dir) 
                        if os.path.isdir(os.path.join(packages_dir, d))]
        
        # Open CSV file for writing
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Package', 'Sourcetype', 'Documentation_URL'])
            
            total_sourcetypes = 0
            
            # Process each package
            for package_name in package_names:
                package_path = os.path.join(packages_dir, package_name)
                
                # Extract documentation URL once per package from README.txt
                
                readme_path = os.path.join(package_path, "README.txt")
                try:
                    with open(readme_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        # Look for URLs (http:// or https://)
                        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                        match = re.search(url_pattern, content)
                        if match:
                            documentation_url =  match.group(0)
                        else:
                            documentation_url = ""
                except FileNotFoundError:
                    pass

                # Extract sourcetypes from props.conf for this package
                sourcetypes = []
                props_conf_path = os.path.join(package_path, "default", "props.conf")
                
                try:
                    with open(props_conf_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    # Find all stanza headers [sourcetype_name]
                    # Match square brackets with content inside, excluding comments
                    pattern = r'^\s*\[([^\]]+)\]\s*$'
                    matches = re.findall(pattern, content, re.MULTILINE)

                    for match in matches:
                        # Clean up the sourcetype name
                        sourcetype = match.strip()
                        # Skip if it contains wildcards or special characters that indicate it's not a sourcetype
                        if not any(char in sourcetype for char in ['*', '?', '...', 'default']):
                            sourcetypes.append(sourcetype)
                
                except FileNotFoundError:
                    print(f"File '{props_conf_path}' not found for package '{package_name}'.")
                
                # Write rows for each sourcetype
                for sourcetype in sourcetypes:
                    writer.writerow([package_name, sourcetype, documentation_url])
                    total_sourcetypes += 1
                
                if sourcetypes:
                    print(f"Package '{package_name}': Found {len(sourcetypes)} sourcetypes")
                else:
                    print(f"Package '{package_name}': No sourcetypes found")

        print(f"Successfully wrote {total_sourcetypes} sourcetypes from {len(package_names)} packages to {csv_filename}")

    except FileNotFoundError:
        print(f"Directory '{packages_dir}' not found.")
        return []

# extract_elastic_logtypes()

extract_splunk_sourcetypes()

