import os
import json
import logging
import csv
from pathlib import Path
from typing import List, Union

import yaml
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def create_embeddings_from_path(
    file_or_folder_path: str,
    collection_name: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = "./chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_batch_size: int = 5461
) -> bool:
    """
    Create embeddings from files in a folder or a single file and store in ChromaDB collection.
    
    Args:
        file_or_folder_path: Path to file or folder to process
        collection_name: Name of the ChromaDB collection to create/use
        embedding_model: HuggingFace embedding model name
        persist_directory: Directory to persist ChromaDB
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        max_batch_size: Maximum number of chunks to process in a single batch
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.yaml', '.yml', '.txt', '.json', '.md', '.conf', '.csv', '.pdf'
    }
    
    def load_yaml_file(file_path: Path) -> List[Document]:
        """Load YAML file and return as Document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Validate YAML
                yaml.safe_load(content)
                return [Document(
                    page_content=content,
                    metadata={
                        'source': file_path.name,
                        'file_type': 'yaml',
                        'full_path': str(file_path)
                    }
                )]
        except Exception as e:
            logging.error(f"Error loading YAML file {file_path}: {e}")
            return []
    
    def load_csv_file(file_path: Path) -> List[Document]:
        """Load CSV file and return as Document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                # Convert CSV to readable text format
                content = "\n".join([",".join(row) for row in rows])
                return [Document(
                    page_content=content,
                    metadata={
                        'source': file_path.name,
                        'file_type': 'csv',
                        'full_path': str(file_path),
                        'rows_count': len(rows)
                    }
                )]
        except Exception as e:
            logging.error(f"Error loading CSV file {file_path}: {e}")
            return []
    
    def load_pdf_file(file_path: Path) -> List[Document]:
        """Load PDF file and return as Document using LangChain's PyPDFLoader"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Update metadata for all pages
            for doc in documents:
                doc.metadata.update({
                    'file_type': 'pdf',
                    'full_path': str(file_path)
                })
            
            return documents
        except Exception as e:
            logging.error(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def load_document(file_path: Path) -> List[Document]:
        """Load a document based on its file type"""
        extension = file_path.suffix.lower()
        
        try:
            if extension in ['.yaml', '.yml']:
                return load_yaml_file(file_path)
            elif extension == '.json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
                documents = loader.load()
                # Update metadata
                for doc in documents:
                    doc.metadata.update({
                        'file_type': 'json',
                        'full_path': str(file_path)
                    })
                return documents
            elif extension == '.csv':
                return load_csv_file(file_path)
            elif extension == '.pdf':
                return load_pdf_file(file_path)
            elif extension in ['.txt', '.md', '.conf']:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                # Update metadata
                for doc in documents:
                    doc.metadata.update({
                        'file_type': extension[1:],  # Remove the dot
                        'full_path': str(file_path)
                    })
                return documents
            else:
                logging.warning(f"Unsupported file type: {extension}")
                return []
                
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            return []
    
    def get_files_to_process(path: Path) -> List[Path]:
        """Get list of files to process from path"""
        files = []
        
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                logging.warning(f"Unsupported file type: {path}")
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(file_path)
        else:
            logging.error(f"Path does not exist: {path}")
            
        return files
    
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Validate input path
        input_path = Path(file_or_folder_path)
        if not input_path.exists():
            logging.error(f"Path does not exist: {file_or_folder_path}")
            return False
        
        # Get files to process
        files_to_process = get_files_to_process(input_path)
        if not files_to_process:
            logging.warning("No supported files found to process")
            return False
        
        logging.info(f"Found {len(files_to_process)} files to process")
        
        # Initialize components
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process files first
        all_documents = []
        for file_path in files_to_process:
            logging.info(f"Processing: {file_path}")
            documents = load_document(file_path)
            all_documents.extend(documents)
        
        if not all_documents:
            logging.warning("No documents were loaded")
            return False
        
        logging.info(f"Loaded {len(all_documents)} documents")
        
        # Split documents into chunks        
        chunks = text_splitter.split_documents(all_documents)
        logging.info(f"Created {len(chunks)} chunks")
        
        # Initialize vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Store chunks using batch processing to handle large files
        total_chunks = len(chunks)
        if total_chunks > max_batch_size:
            logging.info(f"Chunks ({total_chunks}) exceed max batch size ({max_batch_size}). Processing in batches.")
            
            # Process chunks in batches
            for batch_start in range(0, total_chunks, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                batch_num = (batch_start // max_batch_size) + 1
                total_batches = (total_chunks + max_batch_size - 1) // max_batch_size
                
                logging.info(f"Processing batch {batch_num}/{total_batches} "
                           f"(chunks {batch_start + 1}-{batch_end} of {total_chunks})")
                
                try:
                    vectorstore.add_documents(batch_chunks)
                    logging.info(f"Successfully stored batch {batch_num}/{total_batches}")
                except Exception as e:
                    logging.error(f"Error storing batch {batch_num}: {e}")
                    raise
        else:
            # Process all chunks at once if within limit
            logging.info(f"Processing all {total_chunks} chunks in single batch")
            vectorstore.add_documents(chunks)
        
        logging.info(f"Successfully stored {len(chunks)} chunks in collection '{collection_name}'")
        return True
        
    except Exception as e:
        logging.error(f"Error in create_embeddings_from_path: {e}")
        return False

# ============================= Example usage of the functions ========================================
# Query RAG to obtain Package for a specific log type
# active_siem = "elastic"
# logtype = "windows_xml"
# package_prompt = f"Which package/add on do I install to parse {logtype} logs into {active_siem}? Return only the name of the package/add on."

# result, sources = query_rag("elastic_packages", package_prompt)

# Process a single file
success = create_embeddings_from_path("splunk_fields.csv", "splunk_fields")

# Process an entire folder
# success = create_embeddings_from_path("path/to/folder", "folder_collection")

# Use custom parameters
# success = create_embeddings_from_path(
#     file_or_folder_path="data/configs",
#     collection_name="config_docs",
#     chunk_size=2000,
#     chunk_overlap=400
# )