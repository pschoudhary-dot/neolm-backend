'''
This is the original code for the media handler. It has been refactored into smaller, more manageable files. This file is kept for reference and historical purposes.
'''

# import os
# import uuid
# from typing import Dict, List
# from dotenv import load_dotenv
# from supabase import create_client, Client
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import mimetypes
# import torch
# from PIL import Image
# from pydub import AudioSegment
# import cv2
# import tempfile

# # Load environment variables
# load_dotenv()

# # Initialize Supabase client
# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")
# supabase: Client = create_client(url, key)

# # Initialize models with caching
# embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
# whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# # Configuration
# CONFIG = {
#     "bucket_name": "media_files",
#     "temp_dir": os.path.join(os.getcwd(), "media_temp"),
#     "video_frame_interval": 10,  # Extract frame every X seconds
#     "audio_sample_rate": 16000,  # Target sample rate for audio processing
#     "max_text_length": 2000      # Max text length for embedding
# }

# # Check for FFmpeg installation (NEW)
# def check_ffmpeg():
#     try:
#         AudioSegment.ffmpeg.get_ffmpeg_version()
#         return True
#     except:
#         print("\n!!! WARNING: FFmpeg not found - audio processing may fail !!!")
#         print("Install FFmpeg from https://ffmpeg.org/download.html")
#         return False

# check_ffmpeg()

# def get_media_type(file_path: str) -> str:
#     """Determine media type from file extension"""
#     mime_type, _ = mimetypes.guess_type(file_path)
#     if mime_type:
#         if mime_type.startswith('image'):
#             return 'image'
#         elif mime_type.startswith('audio'):
#             return 'audio'
#         elif mime_type.startswith('video'):
#             return 'video'
#     return 'unknown'

# def upload_media(file_path: str) -> str:
#     """Upload media file to Supabase storage (FIXED UPLOAD METHOD)"""
#     try:
#         print(f"\n=== Uploading {file_path} ===")
#         file_name = os.path.basename(file_path)
#         media_type = get_media_type(file_path)
        
#         print(f"│ Media Type: {media_type.title()}")
#         print(f"│ File Name: {file_name}")
        
#         with open(file_path, 'rb') as f:
#             file_content = f.read()
#             # FIX: Changed 'file_path' parameter to 'path'
#             supabase.storage.from_(CONFIG["bucket_name"]).upload(
#                 path=file_name,
#                 file=file_content,
#                 file_options={"content-type": mimetypes.guess_type(file_path)[0]}
#             )
        
#         storage_path = f"{CONFIG['bucket_name']}/{file_name}"
#         print(f"✓ Upload successful: {storage_path}")
#         return storage_path
    
#     except Exception as e:
#         print(f"\n!!! UPLOAD ERROR !!!\n│ {str(e)}")
#         raise

# def process_image(image_path: str) -> str:
#     """Process image file to generate descriptive text"""
#     try:
#         print("│ Processing image...")
#         image = Image.open(image_path)
#         result = image_to_text(image)
#         return result[0]['generated_text']
    
#     except Exception as e:
#         print(f"!!! IMAGE PROCESSING ERROR !!!\n│ {str(e)}")
#         return ""

# def process_audio(audio_path: str) -> str:
#     """Process audio file to extract text"""
#     try:
#         print("│ Processing audio...")
#         audio = AudioSegment.from_file(audio_path)
#         audio = audio.set_frame_rate(CONFIG["audio_sample_rate"])
        
#         with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
#             audio.export(tmpfile.name, format="wav")
#             result = whisper(tmpfile.name)
        
#         return result['text']
    
#     except Exception as e:
#         print(f"!!! AUDIO PROCESSING ERROR !!!\n│ {str(e)}")
#         return ""

# def video_has_audio(video_path: str) -> bool:
#     """Check if video contains audio track (NEW)"""
#     try:
#         video = cv2.VideoCapture(video_path)
#         audio_stream = int(video.get(cv2.CAP_PROP_AUDIO_STREAM))
#         video.release()
#         return audio_stream >= 0
#     except:
#         return False

# def process_video(video_path: str) -> str:
#     """Process video file with audio detection (IMPROVED)"""
#     try:
#         print("│ Processing video...")
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_interval = int(fps * CONFIG["video_frame_interval"])
#         text_descriptions = []
        
#         # Process video frames
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             if frame_count % frame_interval == 0:
#                 with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:
#                     cv2.imwrite(tmpfile.name, frame)
#                     frame_text = process_image(tmpfile.name)
#                     text_descriptions.append(frame_text)
            
#             frame_count += 1
        
#         cap.release()
        
#         # Process audio only if present
#         audio_text = ""
#         if video_has_audio(video_path):
#             print("│ Detected audio track - processing...")
#             try:
#                 video = cv2.VideoCapture(video_path)
#                 audio = AudioSegment.from_file(video_path)
#                 audio = audio.set_frame_rate(CONFIG["audio_sample_rate"])
                
#                 with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
#                     audio.export(tmpfile.name, format="wav")
#                     result = whisper(tmpfile.name)
#                     audio_text = result['text']
#             except Exception as e:
#                 print(f"!!! VIDEO AUDIO ERROR !!!\n│ {str(e)}")
#                 audio_text = "[Audio processing failed]"
#         else:
#             print("│ No audio track detected - skipping audio processing")
#             audio_text = "[No audio present]"
        
#         return " ".join(text_descriptions) + " " + audio_text
    
#     except Exception as e:
#         print(f"!!! VIDEO PROCESSING ERROR !!!\n│ {str(e)}")
#         return ""

# def store_media(media_id: str, media_type: str, text: str, storage_path: str, embedding: List[float]):
#     """Store media metadata in database"""
#     try:
#         print(f"\n=== Storing {media_type} ===")
#         print(f"│ Media ID: {media_id}")
#         print(f"│ Text Length: {len(text)} characters")
#         print(f"│ Embedding Dimensions: {len(embedding)}")
        
#         response = supabase.table('media').insert({
#             "media_id": media_id,
#             "media_type": media_type,
#             "storage_path": storage_path,
#             "description_text": text[:CONFIG["max_text_length"]],
#             "embedding": embedding
#         }).execute()
        
#         if len(response.data) > 0:
#             print(f"✓ {media_type.title()} stored successfully")
#         else:
#             print("!!! STORAGE WARNING !!! No data returned")
    
#     except Exception as e:
#         print(f"\n!!! STORAGE ERROR !!!\n│ {str(e)}")

# def handle_media(file_path: str):
#     """Main media processing handler"""
#     try:
#         os.makedirs(CONFIG["temp_dir"], exist_ok=True)
#         storage_path = upload_media(file_path)
#         media_type = get_media_type(file_path)
        
#         print("│ Downloading for processing...")
#         with open(file_path, 'rb') as f:
#             local_path = os.path.join(CONFIG["temp_dir"], os.path.basename(file_path))
#             with open(local_path, 'wb') as tmpfile:
#                 tmpfile.write(f.read())
        
#         processed_text = ""
#         if media_type == 'image':
#             processed_text = process_image(local_path)
#         elif media_type == 'audio':
#             processed_text = process_audio(local_path)
#         elif media_type == 'video':
#             processed_text = process_video(local_path)
#         else:
#             print("!!! UNSUPPORTED MEDIA TYPE !!!")
#             return
        
#         print(f"│ Processed Text: {processed_text[:200]}...")
#         embedding = embedding_model.encode(processed_text).tolist()
#         media_id = str(uuid.uuid4())
#         store_media(media_id, media_type, processed_text, storage_path, embedding)
#         os.remove(local_path)
    
#     except Exception as e:
#         print(f"\n!!! MEDIA PROCESSING ERROR !!!\n│ {str(e)}")

# def query_media(query_text: str, media_type: str = None, top_k: int = 5) -> List[Dict]:
#     """Search media with optional type filter (FIXED)"""
#     try:
#         print(f"\n=== Searching media: '{query_text}' ===")
#         query_embedding = embedding_model.encode(query_text).tolist()
        
#         # Create the SQL function first if not exists
#         # Execute via Supabase SQL editor:
#         """
#         CREATE OR REPLACE FUNCTION search_media(query_embedding vector(768), match_count int)
#         RETURNS TABLE(media_id uuid, media_type varchar, storage_path text, description_text text, similarity float)
#         AS $$
#         BEGIN
#             RETURN QUERY
#             SELECT
#                 media.media_id,
#                 media.media_type,
#                 media.storage_path,
#                 media.description_text,
#                 1 - (media.embedding <=> query_embedding) AS similarity
#             FROM media
#             ORDER BY media.embedding <=> query_embedding
#             LIMIT match_count;
#         END;
#         $$ LANGUAGE plpgsql;
#         """
        
#         response = supabase.rpc('search_media', {
#             'query_embedding': query_embedding,
#             'match_count': top_k
#         }).execute()
        
#         if media_type:
#             filtered = [r for r in response.data if r['media_type'] == media_type]
#             print(f"✓ Found {len(filtered)} {media_type} results")
#             return filtered[:top_k]
        
#         print(f"✓ Found {len(response.data)} results")
#         return response.data
    
#     except Exception as e:
#         print(f"\n!!! SEARCH ERROR !!!\n│ {str(e)}")
#         return []

# if __name__ == "__main__":
#     try:
#         print("\n=== Processing Example File ===")
#         handle_media("./tmp/voice.mp3")
        
#         print("\n=== Searching for 'spend my life' ===")
#         results = query_media("spend my life", media_type="audio")
        
#         if results:
#             print("\n=== Search Results ===")
#             for idx, result in enumerate(results, 1):
#                 print(f"\n▓ Result #{idx} ({result['media_type'].title()})")
#                 print(f"│ File: {result['storage_path']}")
#                 print(f"│ Similarity: {result['similarity']:.3f}")
#                 print(f"│ Description: {result['description_text'][:200]}...")
#         else:
#             print("No results found")
            
#     except Exception as e:
#         print(f"\n!!! SYSTEM ERROR !!!\n│ {str(e)}")


'''
This is the new AI refractored code that uses centralized config.py file
'''

import os
import uuid
import time
import cv2
import torch
from typing import Dict, List, Optional
from pathlib import Path
import mimetypes
import tempfile
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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

class MediaProcessor:
    def __init__(self):
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.embedding_model = SentenceTransformer(
            Config.DOC["embedding_model"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_pipe = pipeline(
            "image-to-text", 
            model="nlpconnect/vit-gpt2-image-captioning",
            device=0 if torch.cuda.is_available() else -1
        )
        self.audio_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1
        )
        self.temp_files = []

    # def _retry(self, max_retries=3, delay=1.0):
    #     def decorator(func):
    #         def wrapper(*args, **kwargs):
    #             for attempt in range(max_retries):
    #                 try:
    #                     return func(*args, **kwargs)
    #                 except Exception as e:
    #                     if attempt == max_retries - 1:
    #                         raise
    #                     time.sleep(delay * (2 ** attempt))
    #             return None
    #         return wrapper
    #     return decorator

    def _cleanup_temp_files(self):
        for f in self.temp_files:
            try:
                if Path(f).exists():
                    Path(f).unlink()
            except Exception as e:
                print(f"Error cleaning up {f}: {str(e)}")
        self.temp_files = []

    def _get_media_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            for media_type in Config.MEDIA["allowed_types"]:
                if mime_type in Config.MEDIA["allowed_types"][media_type]:
                    return media_type
        raise ValueError(f"Unsupported media type: {mime_type}")

    def _generate_thumbnail(self, frame) -> Optional[bytes]:
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail(Config.MEDIA["thumbnail_size"])
            
            temp_thumb = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(temp_thumb.name)
            self.temp_files.append(temp_thumb.name)
            
            with open(temp_thumb.name, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"Thumbnail generation failed: {str(e)}")
            return None

    def _video_has_audio(self, video_path: str) -> bool:
        try:
            video = cv2.VideoCapture(video_path)
            audio_stream = int(video.get(cv2.CAP_PROP_AUDIO_STREAM))
            video.release()
            return audio_stream >= 0
        except:
            return False

    @_retry(max_retries=3, delay=2)
    def _upload_to_storage(self, file_path: str) -> str:
        try:
            file_name = Path(file_path).name
            mime_type = mimetypes.guess_type(file_path)[0]
            
            with open(file_path, "rb") as f:
                content = f.read()
                self.supabase.storage.from_(Config.MEDIA["bucket_name"]).upload(
                    file_name, content, {"content-type": mime_type}
                )
            
            return f"{Config.MEDIA['bucket_name']}/{file_name}"
        except Exception as e:
            raise RuntimeError(f"Media upload failed: {str(e)}")

    def process_image(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path)
            caption = self.image_pipe(image)[0]["generated_text"]
            return {"description": caption, "thumbnail": None}
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")

    def process_audio(self, audio_path: str) -> Dict:
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(Config.MEDIA["audio_sample_rate"])
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                transcript = self.audio_pipe(tmp.name)["text"]
            
            return {"description": transcript, "thumbnail": None}
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")

    def process_video(self, video_path: str) -> Dict:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * Config.MEDIA["video_frame_interval"])
            descriptions = []
            thumbnail = None
            
            # Capture thumbnail from first valid frame
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    thumbnail = self._generate_thumbnail(frame)
                    break
            
            # Process frames
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                        cv2.imwrite(tmp.name, frame)
                        desc = self.image_pipe(Image.open(tmp.name))[0]["generated_text"]
                        descriptions.append(desc)
                
                frame_count += 1
            
            cap.release()
            
            # Process audio if available
            audio_text = ""
            if self._video_has_audio(video_path):
                try:
                    audio = AudioSegment.from_file(video_path)
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                        audio.export(tmp.name, format="wav")
                        audio_text = self.audio_pipe(tmp.name)["text"]
                except Exception as e:
                    audio_text = f"[Audio Error: {str(e)}]"
            
            return {
                "description": " ".join(descriptions) + " " + audio_text,
                "thumbnail": thumbnail
            }
        except Exception as e:
            raise RuntimeError(f"Video processing failed: {str(e)}")

    def handle_media(self, file_path: str) -> Dict:
        try:
            # Validate file type
            media_type = self._get_media_type(file_path)
            
            # Process media
            if media_type == "image":
                result = self.process_image(file_path)
            elif media_type == "audio":
                result = self.process_audio(file_path)
            elif media_type == "video":
                result = self.process_video(file_path)
            else:
                raise ValueError("Unsupported media type")
            
            # Generate embedding
            embedding = self.embedding_model.encode(
                result["description"][:Config.MEDIA["max_text_length"]]
            ).tolist()
            
            # Upload files
            storage_path = self._upload_to_storage(file_path)
            media_id = str(uuid.uuid4())
            
            # Upload thumbnail if exists
            thumbnail_path = None
            if result["thumbnail"]:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(result["thumbnail"])
                    thumbnail_path = self._upload_to_storage(tmp.name)
            
            # Store metadata
            self.supabase.table("media").insert({
                "media_id": media_id,
                "media_type": media_type,
                "storage_path": storage_path,
                "thumbnail_path": thumbnail_path,
                "description": result["description"],
                "embedding": embedding
            }).execute()
            
            return {
                "status": "success",
                "media_id": media_id,
                "description": result["description"],
                "storage_path": storage_path
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            self._cleanup_temp_files()

    @_retry(max_retries=3, delay=1)
    def query_media(self, query_text: str, media_type: str = None) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()
            response = self.supabase.rpc("search_media", {
                "query_embedding": query_embedding,
                "match_count": Config.MEDIA["max_results"]
            }).execute()
            
            results = response.data if response.data else []
            if media_type:
                results = [r for r in results if r["media_type"] == media_type]
            
            return results[:Config.MEDIA["max_results"]]
        except Exception as e:
            raise RuntimeError(f"Media query failed: {str(e)}")

if __name__ == "__main__":
    processor = MediaProcessor()
    result = processor.handle_media("./tmp/sample.mp4")
    print(result)
    
    if result["status"] == "success":
        query_results = processor.query_media("city skyline at night", media_type="video")
        print("\nQuery Results:", query_results)