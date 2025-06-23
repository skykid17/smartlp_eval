from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

TOP_K = 3

persist_directory = "./chroma_db"

chroma = Chroma(persist_directory=persist_directory)

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")

# Load existing ChromaDB using LangChain's Chroma with the correct collection name
vectorstore = Chroma(
persist_directory=persist_directory,
embedding_function=embeddings,
collection_name="splunk_addons"
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

question = "Which add on do I install the add on to parse windows_xml logs into Splunk? Return only the name of the add on."

qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True
    )

result = qa_chain.invoke({"query": question})
    
print(f"Answer: {result['result']}")
print("\nSource documents:")
for i, doc in enumerate(result['source_documents']):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.page_content[:400]}...")
    if hasattr(doc, 'metadata'):
        print(f"Source: {doc.metadata}")