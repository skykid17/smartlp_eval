import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

TOP_K = 3

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

def query_rag(collection: str, query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", verbose: bool = False):

    embeddings = HuggingFaceEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )
    
    local_llm = ChatOpenAI(
        base_url=endpoint,
        api_key=os.getenv("GITHUB_TOKEN"),
        model=model,
        temperature=0.7
    )

    ## Qwen25 coder 32b instruct is hosted on vLLM on 192.168.125.32:8000
    lab_llm = ChatOpenAI(
        base_url="http://192.168.125.32:8000/v1",
        api_key="EMPTY",
        model="qwen25-coder-32b-awq",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=lab_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    if verbose:
        print(f"Answer: {result['result']}")
        print("\nSource documents:")
        for i, doc in enumerate(result['source_documents']):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:400]}...")
            if hasattr(doc, 'metadata'):
                print(f"Source: {doc.metadata}")

    return result["result"], result["source_documents"]

log = "2025-03-07T11:37:22.109Z server52 DatabaseConnector [INFO]: Operation started for user 520"

query = r'''You are an expert in log parsing and regular expressions. Given a log entry and the siems' 
default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as mmany fields 
as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. 
Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces 
outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only 
the regex pattern. ''' + log

# regex, source = query_rag("elastic_fields", query, verbose=True)
# regex, source = query_rag("splunk_fields", query, verbose=True)