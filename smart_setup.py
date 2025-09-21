# smart_setup.py - Intelligent installation script
import subprocess
import sys
import os

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def test_pdf_libraries():
    """Test which PDF library works"""
    pdf_libraries = [
        ("PyPDF2==3.0.1", "PyPDF2", "from PyPDF2 import PdfReader"),
        ("pypdf==4.0.1", "pypdf", "import pypdf"),
        ("PyMuPDF==1.23.8", "PyMuPDF/fitz", "import fitz")
    ]
    
    for package, name, test_import in pdf_libraries:
        print(f"🔍 Trying {name}...")
        
        # Install the package
        if install_package(package):
            # Test if it imports successfully
            try:
                exec(test_import)
                print(f"✅ {name} installed and working!")
                return name
            except ImportError as e:
                print(f"❌ {name} installed but not working: {e}")
                continue
        else:
            print(f"❌ Failed to install {name}")
    
    return None

def main():
    print("🚀 Smart Resume Relevance System Setup")
    print("=" * 50)
    
    # Core packages that should always work
    core_packages = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0", 
        "sqlalchemy==2.0.23",
        "python-multipart==0.0.6",
        "python-docx==1.1.0",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.2",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "aiofiles==23.2.1"
    ]
    
    print("📦 Installing core packages...")
    failed_packages = []
    
    for package in core_packages:
        package_name = package.split('==')[0]
        print(f"   Installing {package_name}...")
        
        if install_package(package):
            print(f"   ✅ {package_name} installed successfully")
        else:
            print(f"   ❌ Failed to install {package_name}")
            failed_packages.append(package_name)
    
    print(f"\n📄 Installing PDF processing library...")
    pdf_library = test_pdf_libraries()
    
    # Optional packages
    print(f"\n🎯 Installing optional packages...")
    optional_packages = [
        ("spacy", "Better NLP processing (optional)")
    ]
    
    for package, description in optional_packages:
        print(f"   Installing {package} - {description}")
        if install_package(package):
            print(f"   ✅ {package} installed")
            
            # Install spacy model if spacy was installed
            if package == "spacy":
                print("   📚 Installing spaCy English model...")
                try:
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    print("   ✅ spaCy model installed")
                except subprocess.CalledProcessError:
                    print("   ⚠️ spaCy model installation failed (optional)")
        else:
            print(f"   ⚠️ {package} installation failed (optional)")
    
    # Summary
    print(f"\n🎉 Setup Summary")
    print("=" * 30)
    
    if failed_packages:
        print(f"❌ Failed core packages: {', '.join(failed_packages)}")
        print("⚠️ System may not work properly without these packages")
    else:
        print("✅ All core packages installed successfully")
    
    if pdf_library:
        print(f"✅ PDF support: {pdf_library}")
    else:
        print("❌ No PDF library could be installed")
        print("📝 You can still use DOCX and TXT files")
    
    print(f"\n🚀 Next steps:")
    if not failed_packages and pdf_library:
        print("1. Run: python run.py")
        print("2. Open frontend/index.html in your browser")
        print("3. Upload some resumes and job descriptions!")
    else:
        print("1. Fix any failed package installations")
        print("2. For PDF issues, try installing Visual Studio Build Tools")
        print("3. Alternatively, use DOCX format for documents")
    
    # Create a status file
    status = {
        "core_packages_ok": len(failed_packages) == 0,
        "pdf_library": pdf_library,
        "ready_to_run": len(failed_packages) == 0,
        "supported_formats": ["DOCX", "TXT"] + (["PDF"] if pdf_library else [])
    }
    
    print(f"\n💾 Installation status saved to setup_status.txt")
    with open("setup_status.txt", "w") as f:
        f.write("Resume Relevance System Setup Status\n")
        f.write("=" * 40 + "\n")
        f.write(f"Core packages: {'✅ OK' if status['core_packages_ok'] else '❌ Failed'}\n")
        f.write(f"PDF support: {'✅ ' + pdf_library if pdf_library else '❌ None'}\n")
        f.write(f"Ready to run: {'✅ Yes' if status['ready_to_run'] else '❌ No'}\n")
        f.write(f"Supported formats: {', '.join(status['supported_formats'])}\n")

if __name__ == "__main__":
    main()