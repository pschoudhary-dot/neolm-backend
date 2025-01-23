'''
this is the complete code that i wrote initially it upports chumking embeeding and storing and retriving the data using sementic search from the suppabase

'''




# import os
# import uuid
# from typing import List, Dict
# from unstructured.partition.auto import partition
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import pipeline
# import mimetypes
# from keybert import KeyBERT
# from supabase import create_client, Client
# import math  # NEW for progress tracking

# # Load environment variables
# load_dotenv()

# # Initialize Supabase client with proper credentials
# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")
# supabase: Client = create_client(url, key)

# # Initialize models with cache (NEW: added model_kwargs for better performance)
# embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# summarizer = pipeline("summarization", model="t5-small", model_kwargs={"cache_dir": "./model_cache"})
# kw_model = KeyBERT()

# # Constants for configuration (NEW: centralized configuration)
# CONFIG = {
#     "chunk_size": 1000,       # IMPROVED: increased from 500 for better context
#     "chunk_overlap": 200,     # IMPROVED: increased overlap for better continuity
#     "max_summary_length": 150,# Maximum summary length in tokens
#     "min_summary_length": 30, # Minimum summary length
#     "keywords_num": 5,        # Number of keywords to extract
#     "bucket_name": "documents",
#     "temp_dir": os.path.join(os.getcwd(), "temp_files")  # NEW: dedicated temp directory
# }

# def get_mime_type(file_path: str) -> str:
#     """Get proper MIME type with fallback to application/octet-stream"""
#     mime_type, _ = mimetypes.guess_type(file_path)
#     return mime_type or "application/octet-stream"

# def upload_to_storage(file_path: str) -> str:
#     """Upload file to Supabase storage with enhanced debugging"""
#     bucket_name = CONFIG["bucket_name"]
#     file_name = os.path.basename(file_path)
    
#     try:
#         print(f"\n=== Starting upload of {file_path} ===")
#         print(f"│ Bucket: {bucket_name}\n│ File: {file_name}")
        
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File {file_path} not found")

#         mime_type = get_mime_type(file_path)
#         print(f"│ MIME type: {mime_type}")
#         print(f"│ File size: {os.path.getsize(file_path):,} bytes")

#         with open(file_path, 'rb') as f:
#             file_content = f.read()
#             res = supabase.storage.from_(bucket_name).upload(
#                 file_name, 
#                 file_content,
#                 file_options={"content-type": mime_type}
#             )

#         storage_path = f"{bucket_name}/{file_name}"
#         print(f"✓ Successfully uploaded to: {storage_path}")
#         return storage_path

#     except Exception as e:
#         print(f"\n!!! UPLOAD ERROR !!!")
#         print(f"│ File: {file_path}\n│ Error: {str(e)}")
#         raise

# def download_from_storage(storage_path: str) -> str:
#     """Download file from Supabase storage with debugging"""
#     try:
#         print(f"\n=== Downloading {storage_path} ===")
#         bucket, file_name = storage_path.split('/', 1)
        
#         # Create dedicated temp directory
#         os.makedirs(CONFIG["temp_dir"], exist_ok=True)
#         local_path = os.path.join(CONFIG["temp_dir"], file_name)
#         print(f"│ Local path: {local_path}")

#         res = supabase.storage.from_(bucket).download(file_name)
        
#         with open(local_path, 'wb') as f:
#             f.write(res)
        
#         print(f"✓ Downloaded {len(res):,} bytes")
#         return local_path

#     except Exception as e:
#         print(f"\n!!! DOWNLOAD ERROR !!!")
#         print(f"│ Path: {storage_path}\n│ Error: {str(e)}")
#         raise

# def process_document(storage_path: str) -> str:
#     """Process document with error handling"""
#     try:
#         print(f"\n=== Processing document {storage_path} ===")
#         local_path = download_from_storage(storage_path)
        
#         print("│ Partitioning document...")
#         elements = partition(local_path)
#         text = "\n".join([str(el) for el in elements])
        
#         print(f"✓ Extracted {len(text):,} characters")
#         os.remove(local_path)
#         print("│ Cleaned up temporary file")
        
#         return text
    
#     except Exception as e:
#         print(f"\n!!! PROCESSING ERROR !!!")
#         print(f"│ Path: {storage_path}\n│ Error: {str(e)}")
#         return ""

# def chunk_text(text: str) -> List[str]:
#     """Split text with improved chunking strategy"""
#     print("\n=== Chunking text ===")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CONFIG["chunk_size"],
#         chunk_overlap=CONFIG["chunk_overlap"],
#         length_function=len,
#         separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     print(f"✓ Created {len(chunks):,} chunks")
#     print(f"First chunk preview: {chunks[0][:100].replace(chr(10), ' ')}..." if chunks else "No chunks generated")
#     return chunks

# def generate_summary(text: str) -> str:
#     """Generate summary with dynamic length adjustment (NEW)"""
#     try:
#         # Calculate dynamic max_length based on text length
#         word_count = len(text.split())
#         max_length = min(
#             CONFIG["max_summary_length"],
#             max(CONFIG["min_summary_length"], math.floor(word_count * 0.3))
#         )
        
#         print(f"│ Generating summary (max_length={max_length})...")
#         summary = summarizer(text, 
#                            max_length=max_length, 
#                            min_length=CONFIG["min_summary_length"],
#                            do_sample=False)
#         return summary[0]['summary_text']
#     except Exception as e:
#         print(f"!!! SUMMARY ERROR !!!\n│ {str(e)}")
#         return text[:300] + "[TRUNCATED]"  # Fallback

# def extract_keywords(text: str) -> List[str]:
#     """Extract keywords with improved parameters"""
#     try:
#         print("│ Extracting keywords...")
#         keywords = kw_model.extract_keywords(
#             text, 
#             keyphrase_ngram_range=(1, 3),  # IMPROVED: allow 3-word phrases
#             stop_words='english',
#             top_n=CONFIG["keywords_num"],
#             diversity=0.7  # IMPROVED: higher diversity
#         )
#         return [kw[0] for kw in keywords]
#     except Exception as e:
#         print(f"!!! KEYWORD ERROR !!!\n│ {str(e)}")
#         return []

# def store_chunk(chunk_id: str, text: str, summary: str, metadata: Dict, embedding: List[float]):
#     """Store chunk with improved error handling and status tracking"""
#     try:
#         print(f"\n=== Storing chunk {chunk_id} ===")
#         print(f"│ Chunk {metadata['chunk_number']} of {metadata['total_chunks']}")
#         print(f"│ Keywords: {', '.join(metadata['keywords'])}")
#         print(f"│ Summary: {summary[:100]}...")
#         print(f"│ Embedding dimensions: {len(embedding)}")
        
#         response = supabase.table('chunks').insert({
#             "chunk_id": chunk_id,
#             "text": text,
#             "summary": summary,
#             "file_path": metadata["storage_path"],
#             "chunk_number": metadata["chunk_number"],
#             "total_chunks": metadata["total_chunks"],
#             "keywords": metadata["keywords"],
#             "embedding": embedding
#         }).execute()
        
#         if len(response.data) == 0:
#             print("!!! STORAGE WARNING !!! Insert returned no data")
#         else:
#             print(f"✓ Stored chunk {metadata['chunk_number']} successfully")

#     except Exception as e:
#         print(f"\n!!! STORAGE ERROR !!!")
#         print(f"│ Chunk ID: {chunk_id}\n│ Error: {str(e)}")

# def handle_document(file_path: str):
#     """Main document handling with progress tracking (NEW)"""
#     storage_path = upload_to_storage(file_path)
    
#     text = process_document(storage_path)
#     if not text:
#         print(f"\n!!! ABORTING !!! No text extracted from {file_path}")
#         return

#     chunks = chunk_text(text)
#     print(f"\n=== Processing {len(chunks):,} chunks ===")
    
#     for i, chunk in enumerate(chunks):
#         chunk_id = str(uuid.uuid4())
#         print(f"\n● Processing chunk {i+1}/{len(chunks)}")
#         print(f"│ Chunk length: {len(chunk):,} characters")
        
#         # Generate metadata
#         summary = generate_summary(chunk)
#         keywords = extract_keywords(chunk)
#         print(f"│ Generated {len(keywords)} keywords")
        
#         # Generate embedding from full text (IMPROVED: was using summary)
#         embedding = embedding_model.encode(chunk).tolist()  # IMPORTANT CHANGE
        
#         metadata = {
#             "storage_path": storage_path,
#             "chunk_number": i + 1,
#             "total_chunks": len(chunks),
#             "keywords": keywords
#         }

#         store_chunk(chunk_id, chunk, summary, metadata, embedding)
        
#         # Progress indicator
#         if (i+1) % 10 == 0:
#             print(f"\n Processed {i+1}/{len(chunks)} chunks ({((i+1)/len(chunks))*100:.1f}%)")

#     print(f"\n✓ Successfully stored {len(chunks)} chunks from {file_path}")

# def query_chunks(query_text: str, top_k: int = 5) -> List[Dict]:
#     """Enhanced query function with better results handling"""
#     try:
#         print(f"\n=== Querying: '{query_text}' ===")
#         query_embedding = embedding_model.encode(query_text).tolist()
        
#         response = supabase.rpc('search_chunks', {
#             'query_embedding': query_embedding,
#             'match_count': top_k
#         }).execute()
        
#         print(f"✓ Found {len(response.data)} results")
#         return response.data
        
#     except Exception as e:
#         print(f"\n!!! QUERY ERROR !!!\n│ {str(e)}")
#         return []

# if __name__ == "__main__":
#     try:
#         print("=== Document Processing System ===")
#         sample_file = "./tmp/doc.pdf"
        
#         print(f"\n● Source file: {sample_file}")
#         if not os.path.exists(sample_file):
#             raise FileNotFoundError(f"File {sample_file} not found")
#         print(f"✓ File verified ({os.path.getsize(sample_file):,} bytes)")
        
#         handle_document(sample_file)
        
#         # Example query with formatted output
#         print("\n=== Running Example Query ===")
#         query = "large language model evaluation challenges"
#         results = query_chunks(query, top_k=3)
        
#         if results:
#             print("\n=== Top Results ===")
#             for idx, result in enumerate(results, 1):
#                 print(f"\n▓ Result #{idx} (Score: {result['similarity']:.3f})")
#                 print(f"│ File: {result['file_path']}")
#                 print(f"│ Chunk {result['chunk_number']}/{result['total_chunks']}")
#                 print(f"│ Keywords: {', '.join(result['keywords'])}")
#                 print(f"│ Summary: {result['summary']}")
#                 print(f"└ Excerpt: {result['text'][:150]}...")
#         else:
#             print("\n No results found for query")
            
#     except Exception as e:
#         print(f"\n!!! SYSTEM ERROR !!!")
#         print(f"│ Type: {type(e).__name__}\n│ Details: {str(e)}")



'''
this is the code that is improved by the AI using a centralized config.py file and a centralized query function.

'''

import os
import uuid
import time
import math
from typing import List, Dict, Optional
from pathlib import Path
import mimetypes
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from keybert import KeyBERT
from supabase import create_client, Client

from config import Config

load_dotenv()
from functools import wraps
from typing import Callable, TypeVar, Any
T = TypeVar('T')

# Retry decorator for handling transient failures
def _retry(max_retries: int = 3, delay: int = 2) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
            raise last_exception
        return wrapper
    return decorator

class DocProcessor:
    def __init__(self):
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.embedding_model = SentenceTransformer(
            Config.DOC["embedding_model"],
            cache_folder=str(Config.DIRS["cache"])
        )
        self.summarizer = pipeline(
            "summarization", 
            model="t5-small",
            model_kwargs={"cache_dir": str(Config.DIRS["cache"])}
        )
        self.kw_model = KeyBERT()
        self.batch_buffer = []
        self.current_metadata = {}


    @_retry(max_retries=Config.DOC["max_retries"], 
           delay=Config.DOC["retry_delay"])
    def _upload_batch(self):
        if self.batch_buffer:
            response = self.supabase.table('chunks').insert(self.batch_buffer).execute()
            if not response.data:
                raise Exception("Batch insert failed")
            self.batch_buffer.clear()

    def _validate_chunk(self, chunk: str) -> bool:
        return len(chunk.strip()) >= Config.DOC["min_chunk_length"]

    def _get_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    @_retry()
    def upload_to_storage(self, file_path: str) -> str:
        try:
            file_name = Path(file_path).name
            mime_type = self._get_mime_type(file_path)
            
            with open(file_path, 'rb') as f:
                file_content = f.read()
                self.supabase.storage.from_(Config.DOC["bucket_name"]).upload(
                    file_name, 
                    file_content,
                    file_options={"content-type": mime_type}
                )
            
            return f"{Config.DOC['bucket_name']}/{file_name}"
        
        except Exception as e:
            raise RuntimeError(f"Upload failed: {str(e)}") from e

    @_retry()
    def download_from_storage(self, storage_path: str) -> str:
        try:
            bucket, file_name = storage_path.split('/', 1)
            local_path = Config.DIRS["documents"] / file_name
            
            res = self.supabase.storage.from_(bucket).download(file_name)
            with open(local_path, 'wb') as f:
                f.write(res)
            
            return str(local_path)
        
        except Exception as e:
            raise RuntimeError(f"Download failed: {str(e)}") from e

    def process_document(self, file_path: str) -> Optional[str]:
        try:
            elements = partition(file_path)
            return "\n".join(str(el) for el in elements)
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.DOC["chunk_size"],
            chunk_overlap=Config.DOC["chunk_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        return splitter.split_text(text)

    def generate_summary(self, text: str) -> str:
        word_count = len(text.split())
        max_length = min(
            Config.DOC["max_summary_length"],
            max(Config.DOC["min_summary_length"], math.floor(word_count * 0.3))
        )
        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=Config.DOC["min_summary_length"],
            do_sample=False
        )
        return summary[0]['summary_text']

    def extract_keywords(self, text: str) -> List[str]:
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=Config.DOC["keywords_num"],
            diversity=0.7
        )
        return [kw[0] for kw in keywords]

    def _create_chunk_data(self, chunk: str, chunk_num: int, total_chunks: int) -> Dict:
        return {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk,
            "summary": self.generate_summary(chunk),
            "file_path": self.current_metadata["storage_path"],
            "chunk_number": chunk_num + 1,
            "total_chunks": total_chunks,
            "keywords": self.extract_keywords(chunk),
            "embedding": self.embedding_model.encode(chunk).tolist()
        }

    def handle_document(self, file_path: str) -> Dict:
        try:
            self.current_metadata["storage_path"] = self.upload_to_storage(file_path)
            local_path = self.download_from_storage(self.current_metadata["storage_path"])
            
            text = self.process_document(local_path)
            if not text:
                return {"status": "error", "message": "No text extracted"}
            
            chunks = self.chunk_text(text)
            processed_chunks = 0
            
            for idx, chunk in enumerate(chunks):
                if not self._validate_chunk(chunk):
                    continue
                
                chunk_data = self._create_chunk_data(chunk, idx, len(chunks))
                self.batch_buffer.append(chunk_data)
                
                if len(self.batch_buffer) >= Config.DOC["batch_size"]:
                    self._upload_batch()
                    processed_chunks += len(self.batch_buffer)
                
            # Upload remaining chunks
            if self.batch_buffer:
                self._upload_batch()
                processed_chunks += len(self.batch_buffer)
            
            Path(local_path).unlink(missing_ok=True)
            
            return {
                "status": "success",
                "processed_chunks": processed_chunks,
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @_retry()
    def query_chunks(self, query_text: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query_text).tolist()
        response = self.supabase.rpc('search_chunks', {
            'query_embedding': query_embedding,
            'match_count': top_k
        }).execute()
        
        return response.data if response.data else []

if __name__ == "__main__":
    processor = DocProcessor()
    result = processor.handle_document("./tmp/sample.pdf")
    print(result)
    
    if result["status"] == "success":
        query_results = processor.query_chunks("machine learning models", top_k=3)
        print("\nQuery Results:", query_results)