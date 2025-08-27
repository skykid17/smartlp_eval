import hashlib
import os
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Set
import json
import hashlib

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
              persist_directory: str = "./chroma"):

    logging.info(f"Querying RAG with collection: {collection}, query: {query}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection
    )

    ## Qwen25 coder 32b instruct is hosted on vLLM on 192.168.125.31:8000
    llm = ChatOpenAI(
        base_url="http://192.168.125.31:8000/v1",
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
    persist_directory: str = "./rag/chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_batch_size: int = 5461,
    enable_checkpointing: bool = True
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
        enable_checkpointing: Enable checkpointing for resumability
        
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
    
    def get_checkpoint_file_path(collection_name: str, input_path: Path) -> Path:
        """Get the checkpoint file path for a given collection and input path"""
        # Create a unique identifier based on collection name and input path
        path_hash = hashlib.md5(str(input_path.absolute()).encode()).hexdigest()[:8]
        checkpoint_filename = f"checkpoint_{collection_name}_{path_hash}.json"
        return Path(persist_directory) / checkpoint_filename
    
    def load_checkpoint(checkpoint_file: Path) -> Dict:
        """Load checkpoint data from file"""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Loaded checkpoint from: {checkpoint_file}")
                    return data
            except Exception as e:
                logging.warning(f"Error loading checkpoint file {checkpoint_file}: {e}")
        else:
            logging.info(f"No checkpoint file found at: {checkpoint_file}. Starting fresh.")
        return {"processed_subfolders": [], "processed_files": []}
    
    def save_checkpoint(checkpoint_file: Path, processed_subfolders: Set[str], processed_files: Set[str]):
        """Save checkpoint data to file"""
        try:
            checkpoint_data = {
                "processed_subfolders": list(processed_subfolders),
                "processed_files": list(processed_files)
            }
            # Ensure checkpoint directory exists
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.debug(f"Checkpoint saved to: {checkpoint_file}")
        except Exception as e:
            logging.error(f"Error saving checkpoint to {checkpoint_file}: {e}")
    
    def get_subfolders_to_process(path: Path, processed_subfolders: Set[str]) -> List[Path]:
        """Get list of subfolders to process, excluding already processed ones"""
        subfolders = []
        
        if path.is_dir():
            # Get all immediate subfolders
            for subfolder in path.iterdir():
                if subfolder.is_dir():
                    subfolder_key = str(subfolder.relative_to(path))
                    if subfolder_key not in processed_subfolders:
                        subfolders.append(subfolder)
        
        return subfolders
    
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Validate input path
        input_path = Path(file_or_folder_path)
        if not input_path.exists():
            logging.error(f"Path does not exist: {file_or_folder_path}")
            return False
        
        # Initialize checkpointing
        checkpoint_file = None
        processed_subfolders = set()
        processed_files = set()
        
        if enable_checkpointing:
            checkpoint_file = get_checkpoint_file_path(collection_name, input_path)
            checkpoint_data = load_checkpoint(checkpoint_file)
            processed_subfolders = set(checkpoint_data.get("processed_subfolders", []))
            processed_files = set(checkpoint_data.get("processed_files", []))
            
            if processed_subfolders or processed_files:
                logging.info(f"Resuming from checkpoint: {len(processed_subfolders)} subfolders and {len(processed_files)} files already processed")
        
        # Initialize components
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Process based on whether input is a file or directory
        if input_path.is_file():
            # Single file processing
            if str(input_path) not in processed_files:
                logging.info(f"Processing single file: {input_path}")
                documents = load_document(input_path)
                
                if documents:
                    chunks = text_splitter.split_documents(documents)
                    if chunks:
                        vectorstore.add_documents(chunks)
                        logging.info(f"Successfully processed file: {input_path}")
                    
                    if enable_checkpointing and checkpoint_file:
                        processed_files.add(str(input_path))
                        save_checkpoint(checkpoint_file, processed_subfolders, processed_files)
                else:
                    logging.warning(f"No documents loaded from file: {input_path}")
            else:
                logging.info(f"File already processed (from checkpoint): {input_path}")
        
        elif input_path.is_dir():
            # Directory processing with subfolder checkpointing
            subfolders = get_subfolders_to_process(input_path, processed_subfolders)
            
            # If no subfolders exist, process all files in the directory
            if not subfolders:
                # Get all files in the directory (not subfolders)
                files_to_process = []
                for file_path in input_path.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix.lower() in SUPPORTED_EXTENSIONS and
                        str(file_path) not in processed_files):
                        files_to_process.append(file_path)
                
                if files_to_process:
                    logging.info(f"Processing {len(files_to_process)} files in directory (no subfolders)")
                    all_documents = []
                    
                    for file_path in files_to_process:
                        logging.info(f"Processing: {file_path}")
                        documents = load_document(file_path)
                        all_documents.extend(documents)
                        
                        if enable_checkpointing and checkpoint_file:
                            processed_files.add(str(file_path))
                    
                    if all_documents:
                        chunks = text_splitter.split_documents(all_documents)
                        if chunks:
                            # Process chunks with batch handling
                            total_chunks = len(chunks)
                            if total_chunks > max_batch_size:
                                logging.info(f"Processing {total_chunks} chunks in batches of {max_batch_size}")
                                for batch_start in range(0, total_chunks, max_batch_size):
                                    batch_end = min(batch_start + max_batch_size, total_chunks)
                                    batch_chunks = chunks[batch_start:batch_end]
                                    vectorstore.add_documents(batch_chunks)
                            else:
                                vectorstore.add_documents(chunks)
                            
                            logging.info(f"Successfully processed {len(chunks)} chunks from directory")
                    
                    if enable_checkpointing and checkpoint_file:
                        save_checkpoint(checkpoint_file, processed_subfolders, processed_files)
                        
                else:
                    logging.warning("No new files to process in directory")
            else:
                # Process each subfolder separately with checkpointing
                logging.info(f"Found {len(subfolders)} subfolders to process")
                
                for subfolder in subfolders:
                    subfolder_key = str(subfolder.relative_to(input_path))
                    logging.info(f"Processing subfolder: {subfolder_key}")
                    
                    # Get files in this subfolder
                    subfolder_files = []
                    for file_path in subfolder.rglob('*'):
                        if (file_path.is_file() and 
                            file_path.suffix.lower() in SUPPORTED_EXTENSIONS):
                            subfolder_files.append(file_path)
                    
                    if not subfolder_files:
                        logging.info(f"No files found in subfolder: {subfolder_key}")
                        processed_subfolders.add(subfolder_key)
                        continue
                    
                    # Process files in this subfolder
                    subfolder_documents = []
                    for file_path in subfolder_files:
                        logging.info(f"Processing: {file_path}")
                        documents = load_document(file_path)
                        subfolder_documents.extend(documents)
                    
                    if subfolder_documents:
                        # Split and store chunks for this subfolder
                        chunks = text_splitter.split_documents(subfolder_documents)
                        logging.info(f"Created {len(chunks)} chunks from subfolder: {subfolder_key}")
                        
                        if chunks:
                            # Process chunks with batch handling
                            total_chunks = len(chunks)
                            if total_chunks > max_batch_size:
                                logging.info(f"Processing subfolder {subfolder_key}: {total_chunks} chunks in batches")
                                for batch_start in range(0, total_chunks, max_batch_size):
                                    batch_end = min(batch_start + max_batch_size, total_chunks)
                                    batch_chunks = chunks[batch_start:batch_end]
                                    vectorstore.add_documents(batch_chunks)
                            else:
                                vectorstore.add_documents(chunks)
                            
                            logging.info(f"Successfully processed subfolder: {subfolder_key}")
                    
                    # Mark subfolder as processed and save checkpoint
                    processed_subfolders.add(subfolder_key)
                    if enable_checkpointing and checkpoint_file:
                        save_checkpoint(checkpoint_file, processed_subfolders, processed_files)
                        logging.info(f"Checkpoint saved after processing subfolder: {subfolder_key}")
        
        # Clean up checkpoint file if processing completed successfully
        if enable_checkpointing and checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logging.info("Processing completed successfully. Checkpoint file removed.")
            except Exception as e:
                logging.warning(f"Could not remove checkpoint file: {e}")
        
        logging.info(f"Successfully completed processing for collection '{collection_name}'")
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
success = create_embeddings_from_path("./repos/splunk_repo", "splunk_packages")

# Process an entire folder
# success = create_embeddings_from_path("path/to/folder", "folder_collection")

# Use custom parameters
# success = create_embeddings_from_path(
#     file_or_folder_path="data/configs",
#     collection_name="config_docs",
#     chunk_size=2000,
#     chunk_overlap=400
# )