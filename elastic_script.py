#!/usr/bin/env python3
"""
Elastic Packages Documentation Indexer

This script processes and indexes documentation files from Elastic integration packages,
creating embeddings stored in ChromaDB for efficient retrieval and search.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elastic_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomYAMLLoader:
    """Custom YAML loader that returns langchain Document objects"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """Load YAML file and return as Document object"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                yaml_content = file.read()
                
            # Parse YAML to validate it's well-formed
            yaml.safe_load(yaml_content)
            
            # Create Document with full YAML content
            relative_path = os.path.relpath(self.file_path, PACKAGES_DIR)
            document = Document(
                page_content=yaml_content,
                metadata={
                    'source': os.path.basename(self.file_path),
                    'relative_path': relative_path,
                    'file_type': 'yaml'
                }
            )
            return [document]
            
        except Exception as e:
            logger.error(f"Error loading YAML file {self.file_path}: {e}")
            return []

class ElasticPackagesIndexer:
    """Main indexer class for processing Elastic packages documentation"""
    
    def __init__(self, packages_dir: str, persist_dir: str = "./chroma_db"):
        self.packages_dir = Path(packages_dir)
        self.persist_dir = persist_dir
        self.checkpoint_file = "elastic_processed_dirs.json"
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # Initialize embedding model
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection("elastic_packages")
            logger.info("Using existing ChromaDB collection: elastic_packages")
        except:
            self.collection = self.chroma_client.create_collection(
                name="elastic_packages",
                metadata={"description": "Elastic integration packages documentation"}
            )
            logger.info("Created new ChromaDB collection: elastic_packages")
        
        # Load checkpoint
        self.processed_dirs = self.load_checkpoint()
        
        # File extensions to process
        self.supported_extensions = {
            '.json': 'json',
            '.log': 'text',
            '.md': 'text', 
            '.txt': 'text',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.conf': 'text'
        }
        
        # Binary extensions to skip
        self.binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
            '.exe', '.dll', '.so', '.dylib',
            '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.mp3', '.wav', '.mp4', '.avi', '.mov', '.wmv',
            '.bin', '.dat', '.db', '.sqlite', '.rtf'
        }
    
    def load_checkpoint(self) -> set:
        """Load processed directories from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_dirs', []))
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        return set()
    
    def save_checkpoint(self, processed_dir: str):
        """Save processed directory to checkpoint file"""
        self.processed_dirs.add(processed_dir)
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'processed_dirs': list(self.processed_dirs),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on extension"""
        extension = file_path.suffix.lower()
        
        # Skip binary files
        if extension in self.binary_extensions:
            return False
            
        # Process supported extensions
        return extension in self.supported_extensions
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load a single document based on its file type"""
        extension = file_path.suffix.lower()
        file_type = self.supported_extensions.get(extension)
        
        try:
            if file_type == 'json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
                documents = loader.load()
                
            elif file_type == 'text':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                
            elif file_type == 'yaml':
                loader = CustomYAMLLoader(str(file_path))
                documents = loader.load()
                
            else:
                return []
            
            # Add relative path to metadata for all documents
            for doc in documents:
                if 'relative_path' not in doc.metadata:
                    doc.metadata['relative_path'] = str(file_path.relative_to(self.packages_dir))
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path.name
                    
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def process_package_directory(self, package_dir: Path) -> Dict[str, int]:
        """Process all files in a package directory"""
        stats = {'documents': 0, 'chunks': 0, 'files_processed': 0, 'files_skipped': 0}
        documents = []
        
        logger.info(f"Processing package: {package_dir.name}")
        
        # Recursively find all files
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                if self.should_process_file(file_path):
                    docs = self.load_document(file_path)
                    if docs:
                        documents.extend(docs)
                        stats['files_processed'] += 1
                        logger.debug(f"Loaded {len(docs)} documents from {file_path}")
                    else:
                        stats['files_skipped'] += 1
                else:
                    stats['files_skipped'] += 1
        
        stats['documents'] = len(documents)
        
        if documents:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            stats['chunks'] = len(chunks)
            
            # Generate embeddings and store in ChromaDB
            if chunks:
                self.store_chunks_in_chromadb(chunks, package_dir.name)
        
        return stats
    
    def store_chunks_in_chromadb(self, chunks: List[Document], package_name: str):
        """Store document chunks in ChromaDB with embeddings"""
        if not chunks:
            return
            
        # Prepare data for ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{package_name}_{i}_{hash(chunk.page_content[:100])}"
            ids.append(chunk_id)
            
            # Prepare metadata
            metadata = dict(chunk.metadata)
            metadata['package'] = package_name
            metadata['chunk_index'] = i
            metadatas.append(metadata)
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks from {package_name}")
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            logger.info(f"Stored {len(chunks)} chunks from {package_name} in ChromaDB")
            
        except Exception as e:
            logger.error(f"Error storing chunks in ChromaDB for {package_name}: {e}")
    
    def process_all_packages(self):
        """Process all packages in the packages directory"""
        if not self.packages_dir.exists():
            logger.error(f"Packages directory does not exist: {self.packages_dir}")
            return
        
        logger.info(f"Starting processing of packages in: {self.packages_dir}")
        
        # Get all subdirectories (packages)
        package_dirs = [d for d in self.packages_dir.iterdir() if d.is_dir()]
        
        total_stats = {'packages': 0, 'documents': 0, 'chunks': 0, 'files_processed': 0, 'files_skipped': 0}
        
        for package_dir in package_dirs:
            package_rel_path = str(package_dir.relative_to(self.packages_dir))
            
            # Check if already processed
            if package_rel_path in self.processed_dirs:
                logger.info(f"Skipping already processed package: {package_dir.name}")
                continue
            
            try:
                # Process package
                stats = self.process_package_directory(package_dir)
                
                # Update totals
                total_stats['packages'] += 1
                total_stats['documents'] += stats['documents']
                total_stats['chunks'] += stats['chunks']
                total_stats['files_processed'] += stats['files_processed']
                total_stats['files_skipped'] += stats['files_skipped']
                
                # Log package stats
                logger.info(f"Package {package_dir.name} completed: "
                           f"{stats['documents']} documents, {stats['chunks']} chunks, "
                           f"{stats['files_processed']} files processed, "
                           f"{stats['files_skipped']} files skipped")
                
                # Save checkpoint
                self.save_checkpoint(package_rel_path)
                
            except Exception as e:
                logger.error(f"Error processing package {package_dir.name}: {e}")
                continue
        
        # Final summary
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total packages processed: {total_stats['packages']}")
        logger.info(f"Total documents loaded: {total_stats['documents']}")
        logger.info(f"Total chunks created: {total_stats['chunks']}")
        logger.info(f"Total files processed: {total_stats['files_processed']}")
        logger.info(f"Total files skipped: {total_stats['files_skipped']}")
        logger.info("=" * 50)

def main():
    """Main function to run the indexer"""
    # Configuration
    PACKAGES_DIR = r"C:\Users\geola\Documents\GitHub\elastic_repo\packages"
    PERSIST_DIR = "./chroma_db"
    
    # Create indexer and run
    indexer = ElasticPackagesIndexer(PACKAGES_DIR, PERSIST_DIR)
    indexer.process_all_packages()

if __name__ == "__main__":
    # Set global variable for use in CustomYAMLLoader
    PACKAGES_DIR = r"C:\Users\geola\Documents\GitHub\elastic_repo\packages"
    main()