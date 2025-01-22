import os
import uuid
import pg8000
from typing import List, Dict
from unstructured.partition.auto import partition
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from keybert import KeyBERT

# Load environment variables
load_dotenv()

# Initialize PostgreSQL connection
def get_db_connection():
    """
    Establish a connection to the PostgreSQL database using pg8000.
    """
    try:
        conn = pg8000.connect(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT"))
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        exit(1)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "neolm-documents"

# Create or connect to the Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Initialize models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="t5-small")
kw_model = KeyBERT()

def process_document(file_path: str) -> str:
    """
    Process a document and extract text from it.
    Supports CSV, PDF, XLSX, PPT, DOCX, and DOC files.
    """
    try:
        elements = partition(file_path)
        text = "\n".join([str(el) for el in elements])
        return text
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return ""

def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into semantic-aware chunks with overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def generate_summary(text: str) -> str:
    """
    Generate a concise summary of the text chunk.
    """
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error generating summary: {e}")
        return text[:300] + "..."  # Fallback to truncation

def extract_keywords(text: str, n_keywords: int = 5) -> List[str]:
    """
    Extract key keywords/phrases from text.
    """
    try:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', 
                                           top_n=n_keywords, diversity=0.5)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a given text.
    """
    return embedding_model.encode(text).tolist()

def store_in_postgres(chunk_id: str, text: str, summary: str, metadata: Dict):
    """
    Store text and summary in PostgreSQL with enriched metadata.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO chunks (chunk_id, text, summary, file_path, 
                             chunk_number, total_chunks, keywords)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            chunk_id,
            text,
            summary,
            metadata["file_path"],
            metadata["chunk_number"],
            metadata["total_chunks"],
            metadata["keywords"]
        ))
        conn.commit()
    except Exception as e:
        print(f"Error storing text in PostgreSQL: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def store_in_pinecone(chunk_id: str, embeddings: List[float], metadata: Dict):
    """
    Store embeddings in Pinecone with enriched metadata.
    """
    try:
        # Include additional metadata for filtering
        enriched_metadata = {
            "file_path": metadata["file_path"],
            "keywords": metadata["keywords"],
            "summary": metadata["summary"],
            "chunk_number": metadata["chunk_number"],
            "total_chunks": metadata["total_chunks"]
        }
        index.upsert([(chunk_id, embeddings, enriched_metadata)])
    except Exception as e:
        print(f"Error storing embeddings in Pinecone: {e}")

def handle_document(file_path: str):
    """
    Handle document processing with improved chunking and metadata.
    """
    text = process_document(file_path)
    if not text:
        print(f"No text extracted from {file_path}")
        return

    chunks = chunk_text(text)
    print(f"Extracted {len(chunks)} chunks from {file_path}")

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        summary = generate_summary(chunk)
        keywords = extract_keywords(chunk)
        
        metadata = {
            "file_path": file_path,
            "chunk_number": i + 1,
            "total_chunks": len(chunks),
            "keywords": keywords,
            "summary": summary
        }

        # Store in databases
        store_in_postgres(chunk_id, chunk, summary, metadata)
        embeddings = generate_embeddings(summary)  # Embed summary for better relevance
        store_in_pinecone(chunk_id, embeddings, metadata)

    print(f"Stored {len(chunks)} chunks from {file_path}")

def retrieve_text(chunk_id: str, get_summary: bool = True) -> str:
    """
    Retrieve text/summary from PostgreSQL with error handling.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        column = "summary" if get_summary else "text"
        cursor.execute(f"SELECT {column} FROM chunks WHERE chunk_id = %s", (chunk_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error retrieving text from PostgreSQL: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Process document
    sample_file = "doc.pdf"
    handle_document(sample_file)

    # Improved query with metadata filtering
    query_text = "large language model evaluation challenges"
    query_embedding = generate_embeddings(query_text)
    
    query_result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Process results with metadata filtering
    if query_result["matches"]:
        best_match = None
        for match in query_result["matches"]:
            # Prioritize chunks containing keywords
            if "evaluation" in match.metadata.get("keywords", []):
                best_match = match
                break
        
        if not best_match:
            best_match = query_result["matches"][0]

        chunk_id = best_match["id"]
        text = retrieve_text(chunk_id)
        print("Retrieved summary:", text)
        print("Keywords:", best_match.metadata.get("keywords", []))