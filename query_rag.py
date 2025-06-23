from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

TOP_K = 3
SPLUNK_COLLECTION_NAME = "splunk_addons"
ELASTIC_COLLECTION_NAME = "elastic_packages"

SPLUNK_PACKAGE_PROMPT = """Which add on do I install the add on to parse windows_xml logs into Splunk? Return only the name of the add on."""
ELASTIC_PACKAGE_PROMPT = """Which package do I install to parse windows_xml logs into Elasticsearch? Return only the name of the package."""

def query_rag(collection, query):

    persist_directory = "./chroma_db"

    embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")

    # Load existing ChromaDB using LangChain's Chroma with the correct collection name
    vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name=collection
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

    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
            return_source_documents=True
        )

    result = qa_chain.invoke({"query": query})

    print(f"Answer: {result['result']}")
    print("\nSource documents:")
    for i, doc in enumerate(result['source_documents']):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:400]}...")
        if hasattr(doc, 'metadata'):
            print(f"Source: {doc.metadata}")

ACTIVE_SIEM = "elastic"  # Change to "splunk" for Splunk

if ACTIVE_SIEM == "elastic":
    query_rag(ELASTIC_COLLECTION_NAME, ELASTIC_PACKAGE_PROMPT)
else:
    query_rag(SPLUNK_COLLECTION_NAME, SPLUNK_PACKAGE_PROMPT)