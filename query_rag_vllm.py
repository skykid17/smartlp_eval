from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

TOP_K = 3

def query_rag(collection: str, 
              query: str, 
              llm_model: str = "qwen25-coder-32b-awq", 
              embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
              persist_directory: str = "./chroma_db"):

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )

    ## Qwen25 coder 32b instruct is hosted on vLLM on 192.168.125.32:8000
    llm = ChatOpenAI(
        base_url="http://192.168.125.32:8000/v1",
        api_key="EMPTY",
        model=llm_model,
        temperature=0.2
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

    return result["result"], result["source_documents"]

active_siem = "elastic"
logtype = "windows_xml"
package_prompt = f"Which package/add on do I install to parse {logtype} logs into {active_siem}? Return only the name of the package/add on."

if active_siem == "elastic":
    result, sources = query_rag("elastic_packages", package_prompt)
else:
    result, sources = query_rag("splunk_addons", package_prompt)