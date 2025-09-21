# Resume Relevance Check System - Complete Setup Guide

## 🚀 Overview
This is a complete full-stack Resume Relevance Check System that uses FastAPI backend with SQLite database and AI-powered analysis capabilities.

## 📁 Project Structure
```
resume_relevance_system/
├── server.py              # FastAPI backend server
├── database.py            # SQLite database models and configuration
├── models.py              # Pydantic models for API
├── text_processor.py      # AI/NLP processing engine
├── run.py                 # Server startup script
├── startup.py              # Installation script
├── requirements.txt       # Python dependencies
├── index.html            # Frontend web application
└── resume_analysis.db    # SQLite database (created automatically)
```

## ⚡ Quick Start

### 1. Clone/Download Project Files
Create a new directory and save all the backend files from the FastAPI artifact into separate Python files.

### 2. Install Dependencies
```bash
# Option 1: Run the setup script
python setup.py

# Option 2: Install manually
pip install fastapi==0.104.1 uvicorn==0.24.0 sqlalchemy==2.0.23 python-multipart==0.0.6 PyMuPDF==1.23.8 python-docx==1.1.0 sentence-transformers==2.2.2 scikit-learn==1.3.2 python-dotenv==1.0.0 pydantic==2.5.0 aiofiles==23.2.1

# Install spaCy English model (optional but recommended)
python -m spacy download en_core_web_sm
```

### 3. Start the Backend Server
```bash
# Option 1: Using the run script
python run.py

# Option 2: Direct uvicorn command
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

### 4. Open the Frontend
- Save the updated HTML file as `index.html`
- Open `index.html` in your web browser
- The frontend will automatically connect to the backend at `http://127.0.0.1:8000`

## 🔧 Configuration

### Backend Configuration
The system uses SQLite by default. To change to PostgreSQL for production:

```python
# In database.py, replace:
SQLALCHEMY_DATABASE_URL = "sqlite:///./resume_analysis.db"

# With:
SQLALCHEMY_DATABASE_URL = "postgresql://username:password@localhost/resume_analysis"
```

### CORS Configuration
For production, update the CORS settings in `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Database Schema

### Tables Created Automatically:
- **resumes**: Stores uploaded resume files and extracted text
- **job_descriptions**: Stores job description files and text
- **analyses**: Stores analysis results and scores
- **activity_logs**: Tracks user activities and system events

## 🤖 AI/NLP Features

### Text Processing Capabilities:
- **PDF/DOCX Extraction**: Using PyMuPDF and python-docx
- **Hard Match Scoring**: TF-IDF vectorization and keyword matching
- **Soft Match Scoring**: Sentence transformers with cosine similarity
- **Skill Extraction**: NLP-based skill identification
- **Feedback Generation**: Automated personalized recommendations

### Supported File Types:
- **Resumes**: PDF, DOCX
- **Job Descriptions**: PDF, DOCX, TXT

## 🌐 API Endpoints

### File Upload
- `POST /upload/resume/` - Upload resume file
- `POST /upload/job-description/` - Upload job description

### Analysis
- `POST /analyze/` - Analyze resume against job description
- `GET /analyses/` - Get analysis history
- `GET /analyses/{id}` - Get specific analysis details

### Data Management
- `GET /resumes/` - List uploaded resumes
- `GET /job-descriptions/` - List uploaded job descriptions
- `DELETE /resumes/{id}` - Delete resume
- `DELETE /analyses/{id}` - Delete analysis

## 🎯 Usage Workflow

1. **Upload Files**: Upload resume (PDF/DOCX) and job description
2. **Automatic Processing**: Files are processed and text extracted
3. **AI Analysis**: System calculates hard/soft match scores
4. **Results Display**: View detailed analysis with recommendations
5. **History Tracking**: All analyses are saved for future reference

## 🔍 Analysis Metrics

### Scoring System:
- **Hard Match (70% weight)**: Keyword-based matching using TF-IDF
- **Soft Match (30% weight)**: Semantic similarity using embeddings
- **Overall Score**: Weighted combination of hard and soft scores
- **Verdict**: High (≥75%), Medium (50-74%), Low (<50%)

### Features:
- Missing skills identification
- Personalized feedback generation
- Skill gap analysis
- Industry-specific recommendations

## 🛠 Development Mode

### Hot Reload
The server runs in development mode with auto-reload enabled. Changes to Python files will automatically restart the server.

### Debug Mode
Add debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📱 Frontend Features

### Interactive UI:
- Real-time file upload with progress indication
- Animated score visualization
- Responsive design for mobile/desktop
- Search and filter capabilities
- Analysis history dashboard

### Modern Design:
- Gradient backgrounds and glassmorphism effects
- Smooth animations and transitions
- Intuitive user experience
- Professional styling

## 🚨 Troubleshooting

### Common Issues:

1. **CORS Errors**: Make sure backend is running on port 8000
2. **File Upload Fails**: Check file format (PDF/DOCX only)
3. **Analysis Errors**: Verify both files are uploaded successfully
4. **Database Issues**: Delete `resume_analysis.db` to reset database

### Performance Optimization:
- Use PostgreSQL for production workloads
- Implement caching for frequently accessed analyses
- Add database indexing for large datasets
- Consider async processing for large files

## 🌟 Production Deployment

### Backend Deployment:
```bash
# Install production ASGI server
pip install gunicorn

# Run with Gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment:
- Host the HTML file on any web server
- Update API_BASE_URL to production backend URL
- Consider using CDN for static assets

### Database Setup:
- Migrate to PostgreSQL for production
- Set up regular backups
- Implement connection pooling
- Add database monitoring

## 🎉 Success!

Your Resume Relevance Check System is now ready! The system provides:

✅ **AI-Powered Analysis**: Advanced NLP for resume scoring
✅ **Database Integration**: Persistent storage with SQLite/PostgreSQL  
✅ **RESTful API**: Clean, documented API endpoints
✅ **Modern Frontend**: Responsive, interactive web interface
✅ **Production Ready**: Scalable architecture for deployment

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure the backend server is running on port 8000
4. Check browser console for any JavaScript errors


Happy analyzing! 🚀
