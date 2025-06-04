#!/usr/bin/env python3

import os
import sys
import traceback

os.environ["ELASTIC_INTEGRATIONS_PATH"] = r"C:\Users\geola\Documents\GitHub\elastic_integrations\packages"

try:
    # Let's manually check what ChromaDB returns for Apache
    from chroma_client import connect_to_chroma, search_similar_integrations
    from ollama_client import generate_embeddings
    
    connect_to_chroma()
    
    # Create a simple Apache-related query
    apache_description = "Apache HTTP web server access log with GET request, IP address, response code 200, user agent Mozilla"
    
    print(f"Query description: {apache_description}")
    
    # Generate embedding and search
    embedding = generate_embeddings(apache_description)
    results = search_similar_integrations(embedding, top_k=10)
    
    print(f"\nChromaDB returned {len(results)} results:")
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. {result['integration_name']} - Similarity: {result['similarity_score']:.3f}")
        if i < 5 and result['similarity_score'] > 0.5:  # Show description for top matches
            print(f"   Description: {result['description'][:150]}...")
    
    # Check similarity threshold
    from config import SIMILARITY_THRESHOLD
    print(f"\nSimilarity threshold: {SIMILARITY_THRESHOLD}")
    
    high_similarity = [r for r in results if r['similarity_score'] >= SIMILARITY_THRESHOLD]
    print(f"Results above threshold ({SIMILARITY_THRESHOLD}): {len(high_similarity)}")
    
    if high_similarity:
        print("High similarity matches:")
        for r in high_similarity:
            print(f"  - {r['integration_name']}: {r['similarity_score']:.3f}")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
