# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # ========================
    # Supabase Configuration
    # ========================
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # ========================
    # Directory Configuration
    # ========================
    DIRS = {
        "temp_files": Path("temp_files"),        # For document processing
        "media_temp": Path("media_temp"),        # For media processing
        "model_cache": Path("model_cache"),      # For model caching
        "tmp": Path("tmp"),
        "cache": Path("tmp")
        # General temporary files
    }
    
    # ========================
    # Document Processing Configuration
    # ========================
    DOC = {
        "bucket_name": "documents",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "min_chunk_length": 50,
        "max_summary_length": 150,
        "min_summary_length": 30,
        "keywords_num": 5,
        "batch_size": 10,
        "max_retries": 3,
        "retry_delay": 1.5
    }
    
    # ========================
    # Media Processing Configuration
    # ========================
    MEDIA = {
        "bucket_name": "media_files",
        "allowed_types": {
            "image": ["image/jpeg", "image/png", "image/webp"],
            "audio": ["audio/mpeg", "audio/wav", "audio/ogg"],
            "video": ["video/mp4", "video/quicktime", "video/webm"]
        },
        "video": {
            "frame_interval": 10,        # Seconds between frames
            "thumbnail_size": (320, 180),
            "max_duration": 600          # Max video duration in seconds
        },
        "audio": {
            "sample_rate": 16000,
            "max_duration": 300          # Max audio duration in seconds
        },
        "gpu_threshold": 1280,          # Min resolution for GPU processing
        "max_text_length": 2000,
        "max_results": 10               # Max results for media queries
    }
    
    # ========================
    # Web Processing Configuration
    # ========================
    WEB = {
        "table_name": "web_content",
        "user_agents": [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ],
        "request": {
            "delay": 2.0,               # Seconds between requests
            "timeout": 15,              # Request timeout in seconds
            "max_connections": 5        # Max concurrent connections
        },
        "content": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "max_text_length": 10000,
            "reprocess_hours": 24       # Hours before reprocessing same URL
        },
        "sitemap": {
            "paths": ["/sitemap.xml", "/sitemap_index.xml", "/sitemap.txt"],
            "timeout": 10               # Sitemap request timeout
        },
        "render_js": True,              # Enable JavaScript rendering
        "max_results": 10               # Max results for web queries
    }

    # ========================
    # System-wide Configuration
    # ========================
    SYSTEM = {
        "log_level": "INFO",            # Logging level
        "max_file_size": 1024 * 1024 * 100,  # 100MB file size limit
        "cleanup_interval": 3600        # Temp file cleanup interval in seconds
    }

    @classmethod
    def setup(cls):
        """Initialize required directories and environment"""
        # Create all directories
        for dir_path in cls.DIRS.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Set transformers cache
        os.environ["TRANSFORMERS_CACHE"] = str(cls.DIRS["model_cache"])
        
        # Set tempfile defaults
        os.environ["TMPDIR"] = str(cls.DIRS["tmp"])

# Initialize configuration
Config.setup()