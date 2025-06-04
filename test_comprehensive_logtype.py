#!/usr/bin/env python3
"""Comprehensive test script for log type analysis."""

from ollama_client import analyze_log_type
import json

def test_comprehensive_log_types():
    """Test log type analysis with various log formats."""
    
    # Test logs with various formats
    test_logs = [
        {
            "name": "Apache Access Log",
            "log": '127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326 "http://example.com/" "Mozilla/5.0"',
            "expected": "apache_access"
        },
        {
            "name": "Apache Error Log", 
            "log": '[Wed Oct 10 13:55:36.123 2023] [core:error] [pid 1234] AH00121: Request exceeded the limit of 10 internal redirects',
            "expected": "apache_error"
        },
        {
            "name": "Nginx Access Log",
            "log": '192.168.1.1 - user [10/Oct/2023:13:55:36 +0000] "POST /api/login HTTP/1.1" 200 1024 "https://example.com" "curl/7.68.0"',
            "expected": "nginx_access"
        },
        {
            "name": "MySQL Error Log",
            "log": '2023-10-10T13:55:36.123456Z 0 [ERROR] [MY-012345] [Server] Unable to connect to database',
            "expected": "mysql_error"
        },
        {
            "name": "System Log (Linux)",
            "log": 'Oct 10 13:55:36 server systemd[1]: Started Apache HTTP Server',
            "expected": "syslog_linux"
        },
        {
            "name": "Security Log",
            "log": '2023-10-10 13:55:36 Authentication failed for user admin from 192.168.1.100',
            "expected": "security_log"
        },
        {
            "name": "JSON Application Log",
            "log": '{"timestamp":"2023-10-10T13:55:36Z","level":"INFO","message":"User login successful","user_id":"12345"}',
            "expected": "json_application"
        }
    ]
    
    print("Testing comprehensive log type analysis...")
    print("=" * 80)
    
    correct_predictions = 0
    total_tests = len(test_logs)
    
    for test in test_logs:
        print(f"\nTest: {test['name']}")
        print(f"Log: {test['log']}")
        print(f"Expected: {test['expected']}")
        
        try:
            result = analyze_log_type(test['log'])
            predicted = result['log_type']
            print(f"Predicted: {predicted}")
            print(f"Characteristics: {result['characteristics']}")
            print(f"Service: {result['service_name']}")
            
            # Check if prediction matches expected (allowing for variations)
            is_correct = (predicted == test['expected'] or 
                         test['expected'] in predicted or 
                         predicted in test['expected'])
            
            if is_correct:
                correct_predictions += 1
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"Accuracy: {correct_predictions}/{total_tests} ({correct_predictions/total_tests*100:.1f}%)")

if __name__ == "__main__":
    test_comprehensive_log_types()
