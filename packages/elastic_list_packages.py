import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Force the correct path
os.environ["ELASTIC_INTEGRATIONS_PATH"] = r"C:\Users\geola\Documents\GitHub\elastic_integrations\packages"

load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print(f"Looking in: {os.environ['ELASTIC_INTEGRATIONS_PATH']}")
print(f"Path exists: {Path(os.environ['ELASTIC_INTEGRATIONS_PATH']).exists()}")

if Path(os.environ['ELASTIC_INTEGRATIONS_PATH']).exists():
    dirs = [d for d in Path(os.environ['ELASTIC_INTEGRATIONS_PATH']).iterdir() if d.is_dir() and not d.name.startswith('.')]
    for i, d in enumerate(dirs):
        print(f"{d.name}")