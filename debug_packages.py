#!/usr/bin/env python3

import os
from pathlib import Path

# Force the correct path
os.environ["ELASTIC_INTEGRATIONS_PATH"] = r"C:\Users\geola\Documents\GitHub\elastic_integrations\packages"

from integration_discovery import discover_integration_packages
from config import ELASTIC_INTEGRATIONS_PATH
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print(f"Looking in: {ELASTIC_INTEGRATIONS_PATH}")
print(f"Path exists: {ELASTIC_INTEGRATIONS_PATH.exists()}")

if ELASTIC_INTEGRATIONS_PATH.exists():
    dirs = [d for d in ELASTIC_INTEGRATIONS_PATH.iterdir() if d.is_dir() and not d.name.startswith('.')]
    for i, d in enumerate(dirs):
        print(f"  {i+1}. {d.name}")

print("\nDiscovering packages...")
packages = discover_integration_packages()
print(f"Found {len(packages)} packages:")
for p in packages:
    name = p.get('integration_name', 'unknown')
    log_type = p.get('log_type', 'unknown')
    print(f"  - {name} (log type: {log_type})")
