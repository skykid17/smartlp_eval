from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb
import os

TOP_K = 5  # Number of top results to retrieve

def test_rag_system():
    """Test the RAG system with persistent ChromaDB"""
    
    # Set up the persistent directory
    persist_directory = "./chroma_db"
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
      # Load existing ChromaDB using LangChain's Chroma with the correct collection name
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="elastic_packages"
    )
    
    # Initialize the LLM
    llm = OllamaLLM(model="llama3.2")
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True
    )
      # Test question
    question = "How do I install the integration package for parsing windows xml logs into Elasticsearch?"
    
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Get the answer
    result = qa_chain.invoke({"query": question})
    
    print(f"Answer: {result['result']}")
    print("\nSource documents:")
    for i, doc in enumerate(result['source_documents']):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:400]}...")
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
