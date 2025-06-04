"""Setup script for the RAG log analysis system."""

import subprocess
import sys
import time
import requests
from pathlib import Path
import json

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n[{step_num}] {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print_step(2, "Installing Python Dependencies")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install dependencies")
        print(f"   Error: {e.stderr}")
        return False

def check_ollama_service():
    """Check if Ollama service is running."""
    print_step(3, "Checking Ollama Service")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        print("✅ Ollama service is running")
        print(f"   Available models: {len(models)}")
        
        # Check for required models
        required_models = ["llama3.2", "nomic-embed-text"]
        missing_models = []
        
        for required in required_models:
            if not any(required in model for model in model_names):
                missing_models.append(required)
        
        if missing_models:
            print("⚠️  Missing required models:")
            for model in missing_models:
                print(f"   - {model}")
            print("\n   To install missing models, run:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            return False
        else:
            print("✅ All required models are available")
            return True
        
    except requests.exceptions.RequestException:
        print("❌ Ollama service is not running")
        print("   Please start Ollama service:")
        print("   1. Install Ollama from https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. Pull models: ollama pull llama3.2 && ollama pull nomic-embed-text")
        return False

def check_milvus_service():
    """Check if Milvus service is running."""
    print_step(4, "Checking Milvus Service")
    
    try:
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        
        print("✅ Milvus service is accessible")
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print("❌ Milvus service is not accessible")
        print(f"   Error: {e}")
        print("\n   To start Milvus:")
        print("   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest")
        return False

def check_elastic_integrations():
    """Check if Elastic integrations directory exists."""
    print_step(5, "Checking Elastic Integrations")
    
    from config import ELASTIC_INTEGRATIONS_PATH
    
    if not ELASTIC_INTEGRATIONS_PATH.exists():
        print("❌ Elastic integrations directory not found")
        print(f"   Expected path: {ELASTIC_INTEGRATIONS_PATH}")
        print("\n   To clone the repository:")
        print("   git clone https://github.com/elastic/integrations.git")
        print(f"   # Move to: {ELASTIC_INTEGRATIONS_PATH}")
        return False
    
    # Count integration directories
    integration_count = len([d for d in ELASTIC_INTEGRATIONS_PATH.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"✅ Elastic integrations directory found")
    print(f"   Path: {ELASTIC_INTEGRATIONS_PATH}")
    print(f"   Integration packages: {integration_count}")
    return True

def initialize_system_data():
    """Initialize the system with integration data."""
    print_step(6, "Initializing System Data")
    
    try:
        from data_initialization import initialize_integration_data
        
        print("   Starting data initialization (this may take a few minutes)...")
        result = initialize_integration_data(force_refresh=False)
        
        if result["status"] == "success":
            print("✅ System data initialized successfully")
            print(f"   Total integrations: {result.get('total_integrations', 0)}")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            return True
        else:
            print("❌ Data initialization failed")
            print(f"   Error: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print("❌ Data initialization failed")
        print(f"   Error: {e}")
        return False

def run_system_validation():
    """Run final system validation."""
    print_step(7, "Final System Validation")
    
    try:
        from data_initialization import validate_system_setup
        
        results = validate_system_setup()
        
        print("   System Component Status:")
        print(f"   Ollama Available: {'✅' if results['ollama_available'] else '❌'}")
        print(f"   Milvus Available: {'✅' if results['milvus_available'] else '❌'}")
        print(f"   Collection Exists: {'✅' if results['collection_exists'] else '❌'}")
        print(f"   Data Populated: {'✅' if results['data_populated'] else '❌'}")
        print(f"   Total Integrations: {results['total_integrations']}")
        
        system_ready = results.get('system_ready', False)
        
        if system_ready:
            print("\n✅ System is fully operational!")
            return True
        else:
            print("\n❌ System has issues that need to be resolved")
            return False
            
    except Exception as e:
        print("❌ System validation failed")
        print(f"   Error: {e}")
        return False

def run_quick_test():
    """Run a quick test of the system."""
    print_step(8, "Quick System Test")
    
    try:
        from rag_system import analyze_log
        
        test_log = '''127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326'''
        
        print("   Testing with sample Apache log...")
        result = analyze_log(test_log)
        
        print(f"   Result: {result}")
        
        if result and result != "system_error" and result != "processing_error":
            print("✅ System test passed!")
            return True
        else:
            print("❌ System test failed")
            return False
            
    except Exception as e:
        print("❌ System test failed")
        print(f"   Error: {e}")
        return False

def main():
    """Main setup function."""
    print_header("RAG Log Analysis System - Setup")
    
    print("This script will set up and validate the RAG log analysis system.")
    print("Please ensure you have:")
    print("- Ollama installed and running")
    print("- Milvus running (Docker recommended)")
    print("- Elastic integrations repository cloned")
    
    input("\nPress Enter to continue...")
    
    # Setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Ollama Service", check_ollama_service),
        ("Milvus Service", check_milvus_service),
        ("Elastic Integrations", check_elastic_integrations),
        ("System Data", initialize_system_data),
        ("System Validation", run_system_validation),
        ("Quick Test", run_quick_test)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except KeyboardInterrupt:
            print(f"\n\nSetup interrupted by user during: {step_name}")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print_header("Setup Summary")
    
    successful_steps = sum(1 for success in results.values() if success)
    total_steps = len(results)
    
    print(f"Completed: {successful_steps}/{total_steps} steps")
    print("\nStep Results:")
    
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {step_name}: {status}")
    
    if successful_steps == total_steps:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("   1. Run: python cli.py validate")
        print("   2. Run: python cli.py test")
        print("   3. Run: python example_usage.py")
        print("   4. Try: python cli.py analyze -f your_log_file.log")
        return 0
    else:
        print(f"\n⚠️  Setup completed with {total_steps - successful_steps} issues")
        print("Please resolve the failed steps before using the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
