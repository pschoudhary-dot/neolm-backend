import os
import uuid
from typing import List, Dict
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import mimetypes
from keybert import KeyBERT
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client with proper credentials
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
summarizer = pipeline("summarization", model="t5-small")
kw_model = KeyBERT()

def get_mime_type(file_path: str) -> str:
    """Get proper MIME type with fallback to application/octet-stream"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def upload_to_storage(file_path: str) -> str:
    """Upload file to Supabase storage with enhanced debugging"""
    bucket_name = "documents"
    file_name = os.path.basename(file_path)
    
    try:
        print(f"\n=== Starting upload of {file_path} ===")
        print(f"Bucket: {bucket_name}, File: {file_name}")
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        # Get proper MIME type
        mime_type = get_mime_type(file_path)
        print(f"Detected MIME type: {mime_type}")

        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
            print(f"File size: {len(file_content)} bytes")

        # Upload file
        print("Attempting upload...")
        res = supabase.storage.from_(bucket_name).upload(
            file_name, 
            file_content,
            file_options={"content-type": mime_type}
        )
        print("Upload response:", res)

        storage_path = f"{bucket_name}/{file_name}"
        print(f"Successfully uploaded to: {storage_path}")
        return storage_path

    except Exception as e:
        print(f"\n!!! Upload error !!!")
        print(f"File: {file_path}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        raise

def download_from_storage(storage_path: str) -> str:
    """Download file from Supabase storage with debugging"""
    try:
        print(f"\n=== Downloading {storage_path} ===")
        bucket, file_name = storage_path.split('/', 1)
        
        # Windows-compatible temp path
        local_path = os.path.join(os.environ['TEMP'], file_name)
        print(f"Local temp path: {local_path}")

        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        print("Starting download...")
        res = supabase.storage.from_(bucket).download(file_name)
        
        with open(local_path, 'wb') as f:
            f.write(res)
        
        print(f"Downloaded {len(res)} bytes")
        return local_path

    except Exception as e:
        print(f"\n!!! Download error !!!")
        print(f"Storage path: {storage_path}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        raise

def process_document(storage_path: str) -> str:
    """Process document with error handling"""
    try:
        print(f"\n=== Processing document {storage_path} ===")
        local_path = download_from_storage(storage_path)
        
        print("Partitioning document...")
        elements = partition(local_path)
        text = "\n".join([str(el) for el in elements])
        
        print(f"Extracted text length: {len(text)} characters")
        os.remove(local_path)
        print("Cleaned up temp file")
        
        return text
    
    except Exception as e:
        print(f"\n!!! Processing error !!!")
        print(f"Storage path: {storage_path}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        return ""
    
def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text with debugging"""
    print("\n=== Chunking text ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    print("Sample chunk:", chunks[0][:100] + "..." if chunks else "No chunks")
    return chunks

def store_chunk(chunk_id: str, text: str, summary: str, metadata: Dict, embedding: List[float]):
    """Store chunk with debugging"""
    try:
        print(f"\n=== Storing chunk {chunk_id} ===")
        print(f"Chunk number: {metadata['chunk_number']}/{metadata['total_chunks']}")
        print(f"Keywords: {metadata['keywords']}")
        print(f"Embedding length: {len(embedding)}")
        
        response = supabase.table('chunks').insert({
            "chunk_id": chunk_id,
            "text": text,
            "summary": summary,
            "file_path": metadata["storage_path"],
            "chunk_number": metadata["chunk_number"],
            "total_chunks": metadata["total_chunks"],
            "keywords": metadata["keywords"],
            "embedding": embedding
        }).execute()
        
        if len(response.data) == 0:
            print("!!! Insert failed - no data returned !!!")
        else:
            print("Successfully stored chunk")
            
    except Exception as e:
        print(f"\n!!! Storage error !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")

def generate_summary(text: str) -> str:
    """
    Generate a concise summary of the text chunk
    """
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error generating summary: {e}")
        return text[:300] + "..."  # Fallback to truncation

def extract_keywords(text: str, n_keywords: int = 5) -> List[str]:
    """
    Extract key keywords/phrases from text
    """
    try:
        keywords = kw_model.extract_keywords(text, 
                                           keyphrase_ngram_range=(1, 2), 
                                           stop_words='english', 
                                           top_n=n_keywords, 
                                           diversity=0.5)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def store_chunk(chunk_id: str, text: str, summary: str, metadata: Dict, embedding: List[float]):
    """
    Store chunk in Supabase with pgvector
    """
    try:
        response = supabase.table('chunks').insert({
            "chunk_id": chunk_id,
            "text": text,
            "summary": summary,
            "file_path": metadata["storage_path"],
            "chunk_number": metadata["chunk_number"],
            "total_chunks": metadata["total_chunks"],
            "keywords": metadata["keywords"],
            "embedding": embedding
        }).execute()
        
        if len(response.data) == 0:
            print("Failed to insert chunk")
            
    except Exception as e:
        print(f"Error storing chunk: {e}")

def handle_document(file_path: str):
    """
    Handle document processing with Supabase integration
    """
    # Upload to Supabase storage
    storage_path = upload_to_storage(file_path)
    
    # Process document
    text = process_document(storage_path)
    if not text:
        print(f"No text extracted from {file_path}")
        return

    # Chunk text
    chunks = chunk_text(text)
    print(f"Extracted {len(chunks)} chunks from {file_path}")

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        summary = generate_summary(chunk)
        keywords = extract_keywords(chunk)
        embedding = embedding_model.encode(summary).tolist()
        
        metadata = {
            "storage_path": storage_path,
            "chunk_number": i + 1,
            "total_chunks": len(chunks),
            "keywords": keywords
        }

        # Store in Supabase
        store_chunk(chunk_id, chunk, summary, metadata, embedding)

    print(f"Stored {len(chunks)} chunks from {file_path}")

def query_chunks(query_text: str, top_k: int = 3) -> List[Dict]:
    """
    Query chunks using pgvector similarity search
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query_text).tolist()
        
        # Execute vector search
        response = supabase.rpc('search_chunks', {
            'query_embedding': query_embedding,
            'match_count': top_k
        }).execute()
        
        return response.data
        
    except Exception as e:
        print(f"Error querying chunks: {e}")
        return []

if __name__ == "__main__":
    try:
        print("=== Starting document processing ===")
        sample_file = "doc.pdf"
        
        # Verify source file exists
        print(f"\nChecking source file: {sample_file}")
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"Source file {sample_file} not found")
        print("Source file verified")
        
        # Process and store document
        handle_document(sample_file)
        
        # Query documents
        print("\n=== Querying documents ===")
        query = "large language model evaluation challenges"
        print(f"Query: {query}")
        results = query_chunks(query)
        
        # Display results
        if results:
            print(f"\n=== Found {len(results)} results ===")
            for idx, result in enumerate(results):
                print(f"\nResult {idx+1}:")
                print(f"Summary: {result['summary']}")
                print(f"Keywords: {result['keywords']}")
                print(f"Similarity Score: {result['similarity']:.4f}")
                print("---")
        else:
            print("\n!!! No results found !!!")
            
    except Exception as e:
        print(f"\n!!! Fatal error in main process !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")