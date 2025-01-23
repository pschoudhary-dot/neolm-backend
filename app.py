# app.py
import streamlit as st
from web_handler import WebProcessor
from doc_handler import DocProcessor
from media_handler import MediaProcessor
from config import Config
import os

# Page configuration
st.set_page_config(
    page_title="Content Processor",
    page_icon="📄",
    layout="wide"
)

# Initialize processors
web_processor = WebProcessor()
doc_processor = DocProcessor()
media_processor = MediaProcessor()

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a processor:",
    ("Web Content", "Document Processing", "Media Processing")
)

# Main app logic
if option == "Web Content":
    st.title("Web Content Processor")
    
    # URL input
    url = st.text_input("Enter a URL to process:")
    if st.button("Process URL"):
        if url:
            with st.spinner("Processing URL..."):
                result = web_processor.process_url(url)
                if result["status"] == "success":
                    st.success(f"Processed {result['chunks_stored']} chunks from {url}")
                    st.json(result)
                else:
                    st.error(f"Error: {result['message']}")
        else:
            st.warning("Please enter a valid URL.")

    # Query input
    st.subheader("Query Processed Content")
    query = st.text_input("Enter a search query:")
    top_k = st.number_input("Number of results to return:", min_value=1, max_value=10, value=5)
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = web_processor.query_web_content(query, top_k=0.7)
                if results:
                    st.success(f"Found {len(results)} results:")
                    for idx, res in enumerate(results, 1):
                        st.markdown(f"### Result {idx}")
                        st.write(f"**Title:** {res['title']}")
                        st.write(f"**URL:** {res['url']}")
                        st.write(f"**Similarity:** {res['similarity']:.2%}")
                        st.write(f"**Excerpt:** {res['excerpt']}")
                        st.write(f"**Scraped At:** {res['scraped_at']}")
                        st.write("---")
                else:
                    st.warning("No results found.")
        else:
            st.warning("Please enter a search query.")

elif option == "Document Processing":
    st.title("Document Processor")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        # Save the file temporarily
        file_path = os.path.join(Config.DIRS["temp_files"], uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                result = doc_processor.handle_document(file_path)
                if result["status"] == "success":
                    st.success(f"Processed {result['processed_chunks']} chunks from {uploaded_file.name}")
                    st.json(result)
                else:
                    st.error(f"Error: {result['message']}")
        
        # Clean up the temp file
        os.remove(file_path)

    # Query input
    st.subheader("Query Processed Documents")
    query = st.text_input("Enter a search query:")
    top_k = st.number_input("Number of results to return:", min_value=1, max_value=10, value=5)
    if st.button("Search Documents"):
        if query:
            with st.spinner("Searching..."):
                results = doc_processor.query_chunks(query, top_k=top_k)
                if results:
                    st.success(f"Found {len(results)} results:")
                    for idx, res in enumerate(results, 1):
                        st.markdown(f"### Result {idx}")
                        st.write(f"**File Path:** {res['file_path']}")
                        st.write(f"**Chunk {res['chunk_number']} of {res['total_chunks']}")
                        st.write(f"**Keywords:** {', '.join(res['keywords'])}")
                        st.write(f"**Summary:** {res['summary']}")
                        st.write(f"**Excerpt:** {res['text'][:200]}...")
                        st.write("---")
                else:
                    st.warning("No results found.")
        else:
            st.warning("Please enter a search query.")

elif option == "Media Processing":
    st.title("Media Processor")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a media file (Image, Audio, Video):", type=["jpg", "jpeg", "png", "mp3", "wav", "mp4"])
    if uploaded_file is not None:
        # Save the file temporarily
        file_path = os.path.join(Config.DIRS["media_temp"], uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process Media"):
            with st.spinner("Processing media..."):
                result = media_processor.handle_media(file_path)
                if result["status"] == "success":
                    st.success(f"Processed {uploaded_file.name}")
                    st.json(result)
                else:
                    st.error(f"Error: {result['message']}")
        
        # Clean up the temp file
        os.remove(file_path)

    # Query input
    st.subheader("Query Processed Media")
    query = st.text_input("Enter a search query:")
    top_k = st.number_input("Number of results to return:", min_value=1, max_value=10, value=5)
    if st.button("Search Media"):
        if query:
            with st.spinner("Searching..."):
                results = media_processor.query_media(query, top_k=0.7)
                if results:
                    st.success(f"Found {len(results)} results:")
                    for idx, res in enumerate(results, 1):
                        st.markdown(f"### Result {idx}")
                        st.write(f"**Media Type:** {res['media_type']}")
                        st.write(f"**File Path:** {res['storage_path']}")
                        st.write(f"**Description:** {res['description_text'][:200]}...")
                        st.write(f"**Similarity:** {res['similarity']:.2%}")
                        st.write("---")
                else:
                    st.warning("No results found.")
        else:
            st.warning("Please enter a search query.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app allows you to process and query web content, documents, and media files.")