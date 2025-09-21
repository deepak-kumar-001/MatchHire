# run.py - Script to start the server
"""
Simple script to run the FastAPI server
Usage: python run.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )