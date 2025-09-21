# setup.py - Installation script
"""
Setup script for installing required packages
Usage: python setup.py
"""
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "sqlalchemy==2.0.23",
        "python-multipart==0.0.6",
        "PyMuPDF==1.23.8",
        "python-docx==1.1.0",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.2",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "aiofiles==23.2.1"
    ]
    
    for requirement in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"✓ Installed {requirement}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {requirement}")

def install_spacy_model():
    """Install spaCy English model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ Installed spaCy English model")
    except subprocess.CalledProcessError:
        print("✗ Failed to install spaCy model (optional)")

if __name__ == "__main__":
    print("Installing Resume Relevance Check System dependencies...")
    install_requirements()
    install_spacy_model()
    print("\n🎉 Setup complete! Run 'python run.py' to start the server.")