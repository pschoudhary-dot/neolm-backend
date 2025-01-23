'''
This is the original code file 
'''

# import os
# import uuid
# from typing import List, Dict
# from dotenv import load_dotenv
# from supabase import create_client, Client
# from sentence_transformers import SentenceTransformer
# from newspaper import Article, ArticleException
# from datetime import datetime, timedelta, timezone
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load environment variables
# load_dotenv()

# # Initialize Supabase client
# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")
# supabase: Client = create_client(url, key)

# # Initialize models
# embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# # Configuration
# CONFIG = {
#     "table_name": "web_content",
#     "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     "reprocess_hours": 24,
#     "chunk_size": 500,
#     "chunk_overlap": 100,
#     "max_text_length": 10000
# }

# def get_existing_entry(url: str) -> Dict:
#     """Check if URL exists and return latest entry"""
#     try:
#         response = supabase.table(CONFIG["table_name"])\
#                         .select("scraped_at")\
#                         .eq("url", url)\
#                         .order("scraped_at", desc=True)\
#                         .limit(1)\
#                         .execute()
#         return response.data[0] if response.data else None
#     except Exception as e:
#         print(f"Database check error: {str(e)}")
#         return None

# def delete_old_entries(url: str):
#     """Remove existing entries for a URL"""
#     try:
#         response = supabase.table(CONFIG["table_name"])\
#                         .delete()\
#                         .eq("url", url)\
#                         .execute()
#         if response and hasattr(response, 'count') and response.count > 0:
#             print(f"♻️ Removed {response.count} old entries")
#     except Exception as e:
#         print(f"Deletion error: {str(e)}")

# def should_reprocess(url: str) -> bool:
#     """Check if URL needs reprocessing"""
#     existing = get_existing_entry(url)
#     if not existing:
#         return True
    
#     try:
#         # Handle both datetime string and object
#         last_scraped = existing["scraped_at"]
#         if isinstance(last_scraped, str):
#             last_scraped = datetime.fromisoformat(last_scraped.replace('Z', '+00:00')).astimezone(timezone.utc)
        
#         now_utc = datetime.now(timezone.utc)
#         return (now_utc - last_scraped) > timedelta(hours=CONFIG["reprocess_hours"])
#     except Exception as e:
#         print(f"⏳ Time comparison error: {str(e)}")
#         return True

# def store_web_content(content: Dict):
#     """Store chunked content with validation"""
#     try:
#         data = {
#             "chunk_id": str(uuid.uuid4()),
#             "url": content['url'],
#             "title": content['title'],
#             "authors": content['authors'],
#             "chunk_text": content['chunk_text'][:CONFIG["max_text_length"]],
#             "chunk_number": content['chunk_number'],
#             "total_chunks": content['total_chunks'],
#             "embedding": content['embedding'],
#             "scraped_at": datetime.now(timezone.utc).isoformat(timespec='seconds')
#         }
        
#         response = supabase.table(CONFIG["table_name"]).insert(data).execute()
#         return bool(response.data)
#     except Exception as e:
#         print(f"💾 Storage error: {str(e)}")
#         return False

# def handle_web_url(url: str):
#     """Process and store web content with chunking"""
#     try:
#         print(f"\n=== Processing URL: {url} ===")
        
#         if not should_reprocess(url):
#             print(f"⏭️ URL processed within last {CONFIG['reprocess_hours']} hours")
#             return

#         # Delete old entries before processing new
#         delete_old_entries(url)

#         # Download and parse article
#         article = Article(url, 
#                         fetch_images=False, 
#                         browser_user_agent=CONFIG["user_agent"],
#                         request_timeout=15)
#         article.download()
#         article.parse()
#         article.nlp()

#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CONFIG["chunk_size"],
#             chunk_overlap=CONFIG["chunk_overlap"]
#         )
#         chunks = text_splitter.split_text(article.text)
        
#         # Process and store each chunk
#         for idx, chunk in enumerate(chunks):
#             content = {
#                 'url': url,
#                 'title': article.title or "No Title",
#                 'authors': article.authors or ["Unknown"],
#                 'chunk_text': chunk,
#                 'chunk_number': idx + 1,
#                 'total_chunks': len(chunks),
#                 'embedding': embedding_model.encode(chunk).tolist()
#             }
            
#             if store_web_content(content):
#                 print(f"✅ Stored chunk {idx+1}/{len(chunks)}")
#             else:
#                 print(f"❌ Failed to store chunk {idx+1}")

#     except ArticleException as e:
#         print(f"📰 Article error: {str(e)}")
#     except Exception as e:
#         print(f"⚙️ Processing error: {str(e)}")

# def query_web_content(query_text: str, top_k: int = 5) -> List[Dict]:
#     """Search across chunks with deduplication"""
#     try:
#         query_embedding = embedding_model.encode(query_text).tolist()
#         response = supabase.rpc('search_web_content', {
#             'query_embedding': query_embedding,
#             'match_count': top_k * 3  # Get extra for deduplication
#         }).execute()

#         # Deduplicate results by URL chunk
#         unique_results = {}
#         for result in response.data or []:
#             key = f"{result['url']}-{result.get('chunk_number', 0)}"
#             if key not in unique_results:
#                 unique_results[key] = {
#                     'title': result.get('title', 'No Title'),
#                     'url': result.get('url', 'Unknown URL'),
#                     'chunk_text': result.get('chunk_text', '')[:200] + "...",
#                     'similarity': round(result.get('similarity', 0), 4)
#                 }

#         # Return top results by similarity
#         return sorted(unique_results.values(), 
#                      key=lambda x: x['similarity'], 
#                      reverse=True)[:top_k]
#     except Exception as e:
#         print(f"🔍 Query error: {str(e)}")
#         return []

# if __name__ == "__main__":
#     try:
#         # Example usage
#         target_url = "https://www.bbc.com/news/articles/cjw4q7v7ez1o"
#         handle_web_url(target_url)
        
#         # Example queries
#         queries = [
#             "sanctions against Russia",
#             "Ukraine battlefield updates",
#             "diplomatic negotiations"
#         ]
        
#         for query in queries:
#             print(f"\n🔎 Searching for: '{query}'")
#             results = query_web_content(query)
            
#             if results:
#                 print(f"\n=== Top {len(results)} Results ===")
#                 for idx, result in enumerate(results, 1):
#                     print(f"\n🔵 Result {idx}:")
#                     print(f"Title: {result['title']}")
#                     print(f"URL: {result['url']}")
#                     print(f"Relevance: {result['similarity']:.2%}")
#                     print(f"Excerpt: {result['chunk_text']}")
#             else:
#                 print("🚫 No relevant results found")
                
#     except Exception as e:
#         print(f"🔥 Critical error: {str(e)}")


'''
This is the AI refractored code that used config.py
'''

import os
import uuid
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree as ET
from functools import wraps
from typing import Callable, TypeVar, Any
import requests
import bleach
from newspaper import Article, ArticleException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from config import Config

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

class WebProcessor:
    def __init__(self):
        """Initialize the web processor with required clients and models."""
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.embedding_model = SentenceTransformer(Config.DOC["embedding_model"])
        self.rate_limits = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.WEB["user_agents"][0]})

    def _sanitize_content(self, html: str) -> str:
        """Sanitize HTML content to remove potentially harmful elements."""
        return bleach.clean(
            html,
            tags=[],
            attributes={},
            strip=True
        )

    def _check_rate_limit(self, domain: str):
        """Enforce rate limiting for domain requests."""
        now = time.time()
        if domain in self.rate_limits:
            elapsed = now - self.rate_limits[domain]
            if elapsed < Config.WEB["request"]["delay"]:
                sleep_time = Config.WEB["request"]["delay"] - elapsed
                time.sleep(sleep_time)
        self.rate_limits[domain] = now

    def _discover_sitemap(self, base_url: str) -> List[str]:
        """Discover and parse sitemap URLs from a base URL."""
        try:
            sitemap_urls = [
                urljoin(base_url, path) for path in Config.WEB["sitemap"]["paths"]
            ]
            
            for url in sitemap_urls:
                response = self.session.get(url, timeout=Config.WEB["sitemap"]["timeout"])
                if response.status_code == 200:
                    if 'xml' in response.headers['Content-Type']:
                        return self._parse_xml_sitemap(response.text)
                    elif 'text/plain' in response.headers['Content-Type']:
                        return response.text.splitlines()
            return []
        except Exception as e:
            print(f"Sitemap discovery failed: {str(e)}")
            return []

    def _parse_xml_sitemap(self, xml_content: str) -> List[str]:
        """Parse XML sitemap content and extract URLs."""
        try:
            urls = []
            root = ET.fromstring(xml_content)
            
            # Handle sitemap index files
            if root.tag.endswith('sitemapindex'):
                for sitemap in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                    loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
                    urls.extend(self._parse_xml_sitemap(self.session.get(loc).text))
            
            # Handle regular sitemap files
            elif root.tag.endswith('urlset'):
                for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                    loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
                    urls.append(loc)
            
            return urls
        except Exception as e:
            print(f"Sitemap parsing failed: {str(e)}")
            return []

    def _should_reprocess(self, url: str) -> bool:
        """Check if a URL needs to be reprocessed based on last scrape time."""
        try:
            response = self.supabase.table(Config.WEB["table_name"]) \
                .select("scraped_at") \
                .eq("url", url) \
                .order("scraped_at", desc=True) \
                .limit(1) \
                .execute()
            
            if not response.data:
                return True
                
            last_scraped = datetime.fromisoformat(
                response.data[0]["scraped_at"].replace('Z', '+00:00')
            ).astimezone(timezone.utc)
            
            return (datetime.now(timezone.utc) - last_scraped) > \
                timedelta(hours=Config.WEB["content"]["reprocess_hours"])
                
        except Exception as e:
            print(f"Reprocess check failed: {str(e)}")
            return True

    @_retry(max_retries=3, delay=2)
    def _delete_old_entries(self, url: str):
        """Delete old entries for a URL before reprocessing."""
        response = self.supabase.table(Config.WEB["table_name"]) \
            .delete() \
            .eq("url", url) \
            .execute()
        return response.count if response else 0

    def _process_article(self, url: str) -> Optional[Dict]:
        """Process a web article using Newspaper4k."""
        try:
            article = Article(
                url,
                fetch_images=False,
                browser_user_agent=Config.WEB["user_agents"][0],
                request_timeout=Config.WEB["request"]["timeout"]
            )
            article.download()
            article.parse()
            article.nlp()
            
            return {
                "title": article.title,
                "text": self._sanitize_content(article.text),
                "authors": article.authors,
                "keywords": article.keywords,
                "summary": article.summary
            }
        except ArticleException as e:
            print(f"Article processing failed: {str(e)}")
            return None

    def _chunk_content(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's text splitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.WEB["content"]["chunk_size"],
            chunk_overlap=Config.WEB["content"]["chunk_overlap"]
        )
        return splitter.split_text(text)

    @_retry(max_retries=3, delay=1)
    def _store_web_content(self, content: Dict) -> bool:
        """Store processed web content in the database."""
        try:
            data = {
                "chunk_id": str(uuid.uuid4()),
                "url": content["url"],
                "title": content["title"],
                "authors": content["authors"],
                "chunk_text": content["chunk_text"][:Config.WEB["content"]["max_text_length"]],
                "chunk_number": content["chunk_number"],
                "total_chunks": content["total_chunks"],
                "embedding": content["embedding"],
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = self.supabase.table(Config.WEB["table_name"]) \
                .insert(data) \
                .execute()
            
            return bool(response.data)
        except Exception as e:
            print(f"Storage error: {str(e)}")
            return False

    def process_url(self, url: str) -> Dict:
        """Main method to process a URL and store its content."""
        try:
            # Validate and normalize URL
            parsed = urlparse(url)
            if not parsed.scheme:
                url = f"https://{url}"
                parsed = urlparse(url)
                
            domain = parsed.netloc
            self._check_rate_limit(domain)
            
            if not self._should_reprocess(url):
                return {"status": "skipped", "message": "Recent version exists"}
                
            # Delete old entries
            deleted_count = self._delete_old_entries(url)
            print(f"Deleted {deleted_count} old entries for {url}")
            
            # Process main URL
            article_data = self._process_article(url)
            if not article_data:
                return {"status": "error", "message": "Failed to process article"}
                
            # Process sitemap
            sitemap_urls = self._discover_sitemap(url)
            print(f"Found {len(sitemap_urls)} URLs in sitemap")
            
            # Chunk and store content
            chunks = self._chunk_content(article_data["text"])
            stored_chunks = 0
            
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    "url": url,
                    "title": article_data["title"] or "No Title",
                    "authors": article_data["authors"] or ["Unknown"],
                    "chunk_text": chunk,
                    "chunk_number": idx + 1,
                    "total_chunks": len(chunks),
                    "embedding": self.embedding_model.encode(chunk).tolist()
                }
                
                if self._store_web_content(chunk_data):
                    stored_chunks += 1
                
            return {
                "status": "success",
                "url": url,
                "chunks_stored": stored_chunks,
                "sitemap_urls": len(sitemap_urls)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @_retry(max_retries=3, delay=1)
    def query_web_content(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query stored web content using semantic search."""
        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()
            response = self.supabase.rpc("search_web_content", {
                "query_embedding": query_embedding,
                "match_count": top_k * 3  # Overfetch for deduplication
            }).execute()
            
            # Deduplicate by URL and select best chunk
            results = {}
            for item in response.data:
                key = item["url"]
                if key not in results or item["similarity"] > results[key]["similarity"]:
                    results[key] = {
                        "title": item.get("title", "No Title"),
                        "url": item.get("url", "Unknown URL"),
                        "excerpt": item.get("chunk_text", "")[:200] + "...",
                        "similarity": item.get("similarity", 0),
                        "scraped_at": item.get("scraped_at", "Unknown Date")
                    }
            
            return sorted(
                results.values(),
                key=lambda x: x["similarity"],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            print(f"Query failed: {str(e)}")
            return []

if __name__ == "__main__":
    processor = WebProcessor()
    
    # Example usage
    result = processor.process_url("https://www.bbc.com/news/articles/c1ez6313g4po")
    print("Processing result:", result)
    
    if result["status"] == "success":
        query_results = processor.query_web_content("President Donald Trump took office", top_k=3)
        print("\nTop results:")
        for idx, res in enumerate(query_results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {res['title']}")
            print(f"URL: {res['url']}")
            print(f"Similarity: {res['similarity']:.2%}")
            print(f"Excerpt: {res['excerpt']}")