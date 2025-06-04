"""Command-line interface for the RAG log analysis system."""

import argparse
import logging
import json
import sys
from pathlib import Path

from rag_system import analyze_log, get_detailed_analysis, initialize_system
from data_initialization import initialize_integration_data, validate_system_setup, get_integration_statistics
from config import LOG_LEVEL, LOG_FORMAT

def setup_logging(level: str = LOG_LEVEL):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rag_system.log")
        ]
    )

def cmd_analyze_log(args):
    """Command to analyze a log file or content."""
    try:
        if args.file:
            # Read log from file
            log_file = Path(args.file)
            if not log_file.exists():
                print(f"Error: Log file {args.file} does not exist")
                return 1
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
        else:
            # Read log from stdin or prompt
            if sys.stdin.isatty():
                print("Enter log content (Ctrl+D to finish):")
                log_content = sys.stdin.read()
            else:
                log_content = sys.stdin.read()
        
        if not log_content.strip():
            print("Error: No log content provided")
            return 1
        
        if args.detailed:
            # Get detailed analysis
            result = get_detailed_analysis(log_content)
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Detailed analysis saved to {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Get simple recommendation
            recommendation = analyze_log(log_content)
            print(f"Recommended integration: {recommendation}")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing log: {e}")
        return 1

def cmd_initialize_data(args):
    """Command to initialize integration data."""
    try:
        print("Initializing integration data...")
        result = initialize_integration_data(force_refresh=args.force)
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        if 'total_integrations' in result:
            print(f"Total integrations: {result['total_integrations']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        return 0 if result['status'] == 'success' else 1
        
    except Exception as e:
        print(f"Error initializing data: {e}")
        return 1

def cmd_validate_system(args):
    """Command to validate system setup."""
    try:
        print("Validating system setup...")
        results = validate_system_setup()
        
        print("\nSystem Validation Results:")
        print(f"Ollama Available: {'✓' if results['ollama_available'] else '✗'}")
        print(f"ChromaDB Available: {'✓' if results['chroma_available'] else '✗'}")
        print(f"Collection Exists: {'✓' if results['collection_exists'] else '✗'}")
        print(f"Data Populated: {'✓' if results['data_populated'] else '✗'}")
        print(f"Total Integrations: {results['total_integrations']}")
        
        if results.get('last_initialization'):
            print(f"Last Initialization: {results['last_initialization']}")
        
        system_ready = results.get('system_ready', False)
        print(f"\nSystem Ready: {'✓' if system_ready else '✗'}")
        if not system_ready:
            print("\nTo fix issues:")
            if not results['ollama_available']:
                print("- Start Ollama service")
            if not results['chroma_available']:
                print("- Start ChromaDB service or ensure local persistence directory is accessible")
            if not results['data_populated']:
                print("- Run: python cli.py init-data")
        
        return 0
        
    except Exception as e:
        print(f"Error validating system: {e}")
        return 1

def cmd_get_statistics(args):
    """Command to get system statistics."""
    try:
        stats = get_integration_statistics()
        
        if 'error' in stats:
            print(f"Error getting statistics: {stats['error']}")
            return 1
        
        print("Integration Statistics:")
        print(f"Total Integrations: {stats['total_integrations']}")
        print(f"Collection Status: {stats['collection_status']}")
        
        if stats.get('last_update'):
            print(f"Last Update: {stats['last_update']}")
        
        if stats.get('integrations_by_type'):
            print("\nIntegrations by Type:")
            for log_type, count in sorted(stats['integrations_by_type'].items()):
                print(f"  {log_type}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return 1

def cmd_test_system(args):
    """Command to test the system with sample logs."""
    sample_logs = {
        "apache_access": '''127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326 "-" "Mozilla/5.0"''',
        
        "nginx_error": '''2023/10/10 13:55:36 [error] 1234#0: *1 connect() failed (111: Connection refused) while connecting to upstream''',
        
        "syslog": '''Oct 10 13:55:36 server01 sshd[1234]: Failed password for user from 192.168.1.100 port 22 ssh2''',
        
        "windows_event": '''2023-10-10 13:55:36 Application Error 1001 Windows cannot find the specified file.''',
        
        "mysql_error": '''2023-10-10 13:55:36 [ERROR] [MY-010334] [Server] Failed to initialize DD tablespace''',
        
        "docker": '''2023-10-10T13:55:36.123456789Z container_name[1234]: INFO: Application started successfully'''
    }
    
    print("Testing system with sample logs...")
    
    for log_type, log_content in sample_logs.items():
        print(f"\n--- Testing {log_type} ---")
        print(f"Log: {log_content[:80]}...")
        
        try:
            recommendation = analyze_log(log_content)
            print(f"Recommendation: {recommendation}")
        except Exception as e:
            print(f"Error: {e}")
    
    return 0

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="RAG Log Analysis System - Analyze logs and recommend Elastic integrations"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=LOG_LEVEL,
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze log content')
    analyze_parser.add_argument('-f', '--file', help='Log file to analyze')
    analyze_parser.add_argument('-d', '--detailed', action='store_true', help='Get detailed analysis')
    analyze_parser.add_argument('-o', '--output', help='Output file for detailed analysis')
    analyze_parser.set_defaults(func=cmd_analyze_log)
    
    # Initialize data command
    init_parser = subparsers.add_parser('init-data', help='Initialize integration data')
    init_parser.add_argument('--force', action='store_true', help='Force refresh of existing data')
    init_parser.set_defaults(func=cmd_initialize_data)
    
    # Validate system command
    validate_parser = subparsers.add_parser('validate', help='Validate system setup')
    validate_parser.set_defaults(func=cmd_validate_system)
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Get system statistics')
    stats_parser.set_defaults(func=cmd_get_statistics)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test system with sample logs')
    test_parser.set_defaults(func=cmd_test_system)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
