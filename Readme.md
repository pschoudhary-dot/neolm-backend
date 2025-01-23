neolm/
├── media_temp/        # Consider rotating files or adding cleanup mechanism
├── model_cache/       # Good practice for model reuse
├── temp_files/        # Ensure proper cleanup in error cases
├── tmp/               # Should be in .gitignore
├── __init__.py        # Empty? Consider package-level config
├── .env               # Ensure sensitive data isn't committed
├── .gitignore         # Verify all temp dirs are included
├── doc_handler.py     # Core document processing
├── media_handler.py   # Multi-media processing
├── requirements.txt   # Check all dependencies are listed
└── web_handler.py     # Web content pipeline
