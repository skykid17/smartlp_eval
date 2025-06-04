"""Example usage of the RAG log analysis system."""

import logging
from datetime import datetime

from rag_system import analyze_log, get_detailed_analysis
from data_initialization import initialize_integration_data, validate_system_setup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_log_analysis():
    """Demonstrate log analysis with various log types."""
    
    # Sample log entries for testing
    sample_logs = {
        "Apache Access Log": '''127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"''',
        
        "Nginx Error Log": '''2023/10/10 13:55:36 [error] 1234#0: *1 connect() failed (111: Connection refused) while connecting to upstream, client: 192.168.1.100''',
        
        "MySQL Error Log": '''2023-10-10 13:55:36 [ERROR] [MY-010334] [Server] Failed to initialize DD tablespace''',
        
        "Syslog Entry": '''Oct 10 13:55:36 server01 sshd[1234]: Failed password for invalid user admin from 192.168.1.100 port 22 ssh2''',
        
        "Windows Event Log": '''2023-10-10 13:55:36 Application Error 1001 The application failed to initialize properly (0xc0000005).''',
        
        "Docker Container Log": '''2023-10-10T13:55:36.123456789Z webapp-container[1234]: INFO: Starting application server on port 8080''',
        
        "AWS CloudTrail Log": '''{"eventTime":"2023-10-10T13:55:36Z","eventName":"ConsoleLogin","sourceIPAddress":"203.0.113.1","userAgent":"Mozilla/5.0"}''',
        
        "Kubernetes Pod Log": '''2023-10-10 13:55:36 I1010 13:55:36.123456 1 controller.go:123] Successfully synced pod webapp-pod-12345'''
    }
    
    print("=" * 80)
    print("RAG Log Analysis System - Example Usage")
    print("=" * 80)
    
    # Validate system first
    print("\n1. Validating System Setup...")
    validation_results = validate_system_setup()
    
    if not validation_results.get('system_ready', False):
        print("❌ System is not ready!")
        print("Issues found:")
        if not validation_results['ollama_available']:
            print("  - Ollama service not available")
        if not validation_results['milvus_available']:
            print("  - Milvus database not available") 
        if not validation_results['data_populated']:
            print("  - Integration data not populated")
        print("\nPlease run system setup first.")
        return
    else:
        print("✅ System is ready!")
        print(f"   Total integrations loaded: {validation_results['total_integrations']}")
    
    # Analyze each sample log
    print(f"\n2. Analyzing Sample Logs...")
    print("-" * 50)
    
    for i, (log_type, log_content) in enumerate(sample_logs.items(), 1):
        print(f"\n{i}. {log_type}")
        print(f"   Log: {log_content[:100]}...")
        
        try:
            # Get simple recommendation
            start_time = datetime.now()
            recommendation = analyze_log(log_content)
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   Recommendation: {recommendation}")
            print(f"   Analysis time: {analysis_time:.2f}s")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 80)

def example_detailed_analysis():
    """Demonstrate detailed analysis of a log entry."""
    
    print("\n3. Detailed Analysis Example")
    print("-" * 50)
    
    # Use a more complex log for detailed analysis
    complex_log = '''
    192.168.1.100 - - [10/Oct/2023:13:55:36 +0000] "POST /api/login HTTP/1.1" 401 157 "https://example.com/login" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    192.168.1.100 - - [10/Oct/2023:13:55:37 +0000] "POST /api/login HTTP/1.1" 401 157 "https://example.com/login" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    192.168.1.100 - - [10/Oct/2023:13:55:38 +0000] "POST /api/login HTTP/1.1" 200 2048 "https://example.com/login" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    '''
    
    try:
        print("Performing detailed analysis...")
        result = get_detailed_analysis(complex_log)
        
        if result['status'] == 'success':
            print(f"\n📊 Analysis Results:")
            print(f"   Log Type: {result['log_analysis']['log_type']}")
            print(f"   Confidence: {result['log_analysis']['confidence']:.2f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            
            if result['recommendations']:
                print(f"\n🎯 Top Recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"   {i}. {rec['integration_name']}")
                    print(f"      Score: {rec['final_score']:.3f}")
                    print(f"      Confidence: {rec['confidence_level']}")
                    print(f"      Factors: {', '.join(rec['scoring_factors'])}")
            else:
                print("\n❌ No recommendations found")
            
            print(f"\n🔍 Log Metadata:")
            metadata = result['preprocessed_log']['metadata']
            print(f"   IP Addresses: {len(metadata.get('ip_addresses', []))}")
            print(f"   Status Codes: {metadata.get('status_codes', [])}")
            print(f"   Timestamps: {len(metadata.get('timestamps', []))}")
            print(f"   Token Count: {result['preprocessed_log']['token_count']}")
            
        else:
            print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")

def main():
    """Run example usage demonstrations."""
    try:
        example_log_analysis()
        example_detailed_analysis()
        
        print("\n" + "=" * 80)
        print("Example completed! To use the system:")
        print("1. Run: python cli.py validate  # Check system status")
        print("2. Run: python cli.py init-data # Initialize data if needed")
        print("3. Run: python cli.py analyze -f your_log_file.log")
        print("4. Run: python cli.py test  # Test with sample logs")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    main()
