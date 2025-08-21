#!/usr/bin/env python3
"""
Production-ready FastAPI app with bundled ChromaDB for deployment
"""
import os
import sys
from pathlib import Path

# Add src to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.main import app

# Ensure ChromaDB directory exists and is writable
chroma_db_path = current_dir / "chroma_db"
chroma_db_path.mkdir(exist_ok=True)

# Set ChromaDB path for the application
os.environ["CHROMA_DB_PATH"] = str(chroma_db_path)

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Legal Assistant API on {host}:{port}")
    print(f"📁 ChromaDB path: {chroma_db_path}")
    
    uvicorn.run(
        "app:app",  # Use string reference for better compatibility
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False
    )
