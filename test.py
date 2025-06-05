from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb
import os

def test_rag_system():
    """Test the RAG system with persistent ChromaDB"""
    
    # Set up the persistent directory
    persist_directory = "chroma_db"
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
      # Load existing ChromaDB using LangChain's Chroma with the correct collection name
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="elastic_packages"
    )
    
    # Debug: Check if the vectorstore has any documents using the underlying client
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(name="elastic_packages")
        count = collection.count()
        print(f"Number of documents in ChromaDB: {count}")
        
        if count > 0:
            # Get a sample of documents to see what's stored
            sample_docs = collection.peek(limit=3)
            print(f"Sample documents:")
            for i, doc in enumerate(sample_docs['documents']):
                print(f"Doc {i+1}: {doc[:100]}...")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        # Try the elastic_packages collection
        try:
            collection = client.get_collection(name="elastic_packages")
            count = collection.count()
            print(f"Number of documents in elastic_packages collection: {count}")
            
            if count > 0:
                sample_docs = collection.peek(limit=3)
                print(f"Sample documents from elastic_packages:")
                for i, doc in enumerate(sample_docs['documents']):
                    print(f"Doc {i+1}: {doc[:100]}...")
        except Exception as e2:
            print(f"Error accessing elastic_packages collection: {e2}")
    
    # Initialize the LLM
    llm = OllamaLLM(model="llama3.2")
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
      # Test question
    question = "How do I install the integration package for parsing windows xml logs into Elasticsearch?"
    
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Debug: Test the retriever directly
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    print(f"Number of documents retrieved: {len(retrieved_docs)}")
    
    # Get the answer
    result = qa_chain.invoke({"query": question})
    
    print(f"Answer: {result['result']}")
    print("\nSource documents:")
    for i, doc in enumerate(result['source_documents']):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata'):
            print(f"Source: {doc.metadata}")
    
    return result

if __name__ == "__main__":
    try:
        result = test_rag_system()
        print("\n" + "="*50)
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
