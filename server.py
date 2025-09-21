# server.py - Updated with enhanced text processing
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import json
from datetime import datetime

from database import get_db, Resume, JobDescription, Analysis, ActivityLog
from models import AnalysisResponse, AnalysisResult

# Use the enhanced text processor
try:
    from text_processor import EnhancedTextProcessor
    text_processor = EnhancedTextProcessor()
    print("✅ Using Enhanced Text Processor with optimized skills matching")
except ImportError:
    from text_processor import TextProcessor
    text_processor = TextProcessor()
    print("⚠️ Using Basic Text Processor")

app = FastAPI(
    title="Resume Relevance Check API", 
    version="2.0.0",
    description="AI-Powered Resume Analysis with Enhanced Matching"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("🚀 Resume Relevance Check API v2.0 is starting up...")
    print(f"📊 PDF Library: {getattr(text_processor, 'PDF_LIBRARY', 'Unknown')}")
    print(f"🧠 Skills Database: {len(getattr(text_processor, 'technical_skills', []))} skills loaded")

@app.get("/")
async def root():
    return {
        "message": "Resume Relevance Check API v2.0", 
        "status": "running",
        "features": ["Enhanced Skills Matching", "Multi-PDF Support", "Industry-Specific Analysis"],
        "pdf_support": hasattr(text_processor, 'PDF_LIBRARY')
    }

@app.post("/upload/resume/")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process resume file with enhanced text extraction"""
    
    # Validate file type
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF and DOCX files are supported. Please convert your file to one of these formats."
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            try:
                extracted_text = text_processor.extract_text_from_pdf(file_content)
                file_type = "pdf"
            except Exception as pdf_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"PDF processing failed: {str(pdf_error)}. Please try converting to DOCX format."
                )
        else:
            extracted_text = text_processor.extract_text_from_docx(file_content)
            file_type = "docx"
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from the file. Please ensure the file contains readable text."
            )
        
        # Clean text with enhanced processing
        cleaned_text = text_processor.clean_text(extracted_text)
        
        # Extract skills for preview
        if hasattr(text_processor, 'extract_skills_enhanced'):
            skills_found = text_processor.extract_skills_enhanced(cleaned_text)
            skills_preview = f" | Skills detected: {', '.join(skills_found[:5])}{'...' if len(skills_found) > 5 else ''}"
        else:
            skills_preview = ""
        
        # Save to database
        db_resume = Resume(
            filename=file.filename,
            file_type=file_type,
            file_content=file_content,
            extracted_text=cleaned_text
        )
        db.add(db_resume)
        db.commit()
        db.refresh(db_resume)
        
        # Log activity with more details
        log_entry = ActivityLog(
            action="upload_resume",
            details=f"Uploaded resume: {file.filename} | Type: {file_type} | Text length: {len(cleaned_text)} chars{skills_preview}"
        )
        db.add(log_entry)
        db.commit()
        
        # Create enhanced preview
        preview_text = cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        
        return {
            "message": "Resume uploaded and processed successfully",
            "resume_id": db_resume.id,
            "filename": file.filename,
            "file_type": file_type,
            "extracted_text_preview": preview_text + skills_preview,
            "text_length": len(cleaned_text),
            "processing_status": "enhanced" if hasattr(text_processor, 'extract_skills_enhanced') else "basic"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/upload/job-description/")
async def upload_job_description(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process job description file with enhanced text extraction"""
    
    allowed_types = [
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
        "text/plain"
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF, DOCX, and TXT files are supported for job descriptions."
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            try:
                extracted_text = text_processor.extract_text_from_pdf(file_content)
                file_type = "pdf"
            except Exception as pdf_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"PDF processing failed: {str(pdf_error)}. Please try converting to DOCX or TXT format."
                )
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = text_processor.extract_text_from_docx(file_content)
            file_type = "docx"
        else:
            extracted_text = file_content.decode('utf-8')
            file_type = "txt"
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < 30:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from the job description. Please check the file content."
            )
        
        # Clean text with enhanced processing
        cleaned_text = text_processor.clean_text(extracted_text)
        
        # Extract required skills for preview
        if hasattr(text_processor, 'extract_skills_enhanced'):
            required_skills = text_processor.extract_skills_enhanced(cleaned_text)
            skills_preview = f" | Required skills: {', '.join(required_skills[:5])}{'...' if len(required_skills) > 5 else ''}"
        else:
            skills_preview = ""
        
        # Save to database
        db_job = JobDescription(
            filename=file.filename,
            file_type=file_type,
            file_content=file_content,
            extracted_text=cleaned_text
        )
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        
        # Log activity
        log_entry = ActivityLog(
            action="upload_job_description",
            details=f"Uploaded job description: {file.filename} | Type: {file_type} | Text length: {len(cleaned_text)} chars{skills_preview}"
        )
        db.add(log_entry)
        db.commit()
        
        # Create enhanced preview
        preview_text = cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        
        return {
            "message": "Job description uploaded and processed successfully",
            "job_description_id": db_job.id,
            "filename": file.filename,
            "file_type": file_type,
            "extracted_text_preview": preview_text + skills_preview,
            "text_length": len(cleaned_text),
            "processing_status": "enhanced" if hasattr(text_processor, 'extract_skills_enhanced') else "basic"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job description: {str(e)}")

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_resume(resume_id: int = Form(...), job_description_id: int = Form(...), db: Session = Depends(get_db)):
    """Analyze resume against job description with enhanced AI processing"""
    
    # Get resume and job description from database
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    job_desc = db.query(JobDescription).filter(JobDescription.id == job_description_id).first()
    
    if not resume or not job_desc:
        raise HTTPException(status_code=404, detail="Resume or job description not found")
    
    try:
        # Use enhanced analysis if available
        if hasattr(text_processor, 'calculate_enhanced_hard_score'):
            print("🔍 Running enhanced analysis...")
            
            # Enhanced hard score calculation
            hard_score, missing_skills = text_processor.calculate_enhanced_hard_score(
                resume.extracted_text, job_desc.extracted_text
            )
            
            # Enhanced semantic similarity
            soft_score = text_processor.calculate_semantic_score(
                resume.extracted_text, job_desc.extracted_text
            )
            
            # Enhanced feedback generation
            feedback = text_processor.generate_enhanced_feedback(
                hard_score, soft_score, missing_skills, 
                resume.extracted_text, job_desc.extracted_text
            )
        else:
            print("⚠️ Using basic analysis...")
            
            # Fallback to basic analysis
            hard_score, missing_skills = text_processor.calculate_hard_match_score(
                resume.extracted_text, job_desc.extracted_text
            )
            soft_score = text_processor.calculate_soft_match_score(
                resume.extracted_text, job_desc.extracted_text
            )
            feedback = text_processor.generate_feedback(hard_score, soft_score, missing_skills)
        
        # Calculate overall score (70% hard, 30% soft)
        overall_score = (hard_score * 0.7) + (soft_score * 0.3)
        
        # Determine verdict with enhanced thresholds
        if overall_score >= 75:
            verdict = "High"
        elif overall_score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        # Save analysis to database
        db_analysis = Analysis(
            resume_id=resume_id,
            job_description_id=job_description_id,
            hard_score=hard_score,
            soft_score=soft_score,
            overall_score=overall_score,
            verdict=verdict,
            missing_skills=json.dumps(missing_skills),
            feedback=feedback
        )
        db.add(db_analysis)
        db.commit()
        
        # Log detailed analysis activity
        log_entry = ActivityLog(
            action="analyze",
            details=f"Enhanced analysis completed | Resume: {resume.filename} vs Job: {job_desc.filename} | Score: {overall_score:.1f}% ({verdict}) | Missing skills: {len(missing_skills)}"
        )
        db.add(log_entry)
        db.commit()
        
        return AnalysisResponse(
            hard_score=round(hard_score, 2),
            soft_score=round(soft_score, 2),
            overall_score=round(overall_score, 2),
            verdict=verdict,
            missing_skills=missing_skills,
            feedback=feedback
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during enhanced analysis: {str(e)}")

@app.get("/analyses/", response_model=List[AnalysisResult])
async def get_analyses(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    """Get analysis history with enhanced details"""
    analyses = db.query(Analysis).join(Resume).offset(skip).limit(limit).all()
    
    results = []
    for analysis in analyses:
        resume = db.query(Resume).filter(Resume.id == analysis.resume_id).first()
        results.append(AnalysisResult(
            id=analysis.id,
            resume_filename=resume.filename,
            hard_score=analysis.hard_score,
            soft_score=analysis.soft_score,
            overall_score=analysis.overall_score,
            verdict=analysis.verdict,
            missing_skills=json.loads(analysis.missing_skills),
            feedback=analysis.feedback,
            analysis_date=analysis.analysis_date
        ))
    
    return results

@app.get("/analyses/{analysis_id}")
async def get_analysis_details(analysis_id: int, db: Session = Depends(get_db)):
    """Get detailed analysis by ID with enhanced information"""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    resume = db.query(Resume).filter(Resume.id == analysis.resume_id).first()
    job_desc = db.query(JobDescription).filter(JobDescription.id == analysis.job_description_id).first()
    
    return {
        "id": analysis.id,
        "resume_filename": resume.filename,
        "job_description_filename": job_desc.filename,
        "hard_score": analysis.hard_score,
        "soft_score": analysis.soft_score,
        "overall_score": analysis.overall_score,
        "verdict": analysis.verdict,
        "missing_skills": json.loads(analysis.missing_skills),
        "feedback": analysis.feedback,
        "analysis_date": analysis.analysis_date,
        "processing_type": "enhanced" if hasattr(text_processor, 'calculate_enhanced_hard_score') else "basic"
    }

@app.get("/resumes/")
async def get_resumes(db: Session = Depends(get_db)):
    """Get all uploaded resumes with enhanced details"""
    resumes = db.query(Resume).all()
    return [
        {
            "id": r.id, 
            "filename": r.filename, 
            "file_type": r.file_type,
            "upload_date": r.upload_date,
            "text_length": len(r.extracted_text) if r.extracted_text else 0
        } for r in resumes
    ]

@app.get("/job-descriptions/")
async def get_job_descriptions(db: Session = Depends(get_db)):
    """Get all uploaded job descriptions with enhanced details"""
    job_descs = db.query(JobDescription).all()
    return [
        {
            "id": j.id, 
            "filename": j.filename, 
            "file_type": j.file_type,
            "upload_date": j.upload_date,
            "text_length": len(j.extracted_text) if j.extracted_text else 0
        } for j in job_descs
    ]

@app.get("/system/status")
async def get_system_status():
    """Get system status and capabilities"""
    return {
        "api_version": "2.0.0",
        "enhanced_processing": hasattr(text_processor, 'calculate_enhanced_hard_score'),
        "pdf_support": hasattr(text_processor, 'extract_text_from_pdf'),
        "pdf_library": getattr(text_processor, 'PDF_LIBRARY', 'None'),
        "skills_database_size": len(getattr(text_processor, 'technical_skills', [])),
        "supported_formats": ["PDF", "DOCX", "TXT"],
        "features": [
            "Enhanced Skills Matching",
            "Industry-Specific Analysis", 
            "Multi-Library PDF Support",
            "Semantic Similarity Analysis",
            "Personalized Feedback Generation"
        ]
    }

@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    """Delete a resume and associated analyses"""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Delete associated analyses
    db.query(Analysis).filter(Analysis.resume_id == resume_id).delete()
    
    # Delete resume
    db.delete(resume)
    db.commit()
    
    # Log deletion
    log_entry = ActivityLog(
        action="delete_resume",
        details=f"Deleted resume: {resume.filename} and associated analyses"
    )
    db.add(log_entry)
    db.commit()
    
    return {"message": "Resume and associated analyses deleted successfully"}

@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Delete a specific analysis"""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    db.delete(analysis)
    db.commit()
    
    # Log deletion
    log_entry = ActivityLog(
        action="delete_analysis",
        details=f"Deleted analysis ID: {analysis_id}"
    )
    db.add(log_entry)
    db.commit()
    
    return {"message": "Analysis deleted successfully"}

@app.get("/analytics/summary")
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Get system usage analytics"""
    total_resumes = db.query(Resume).count()
    total_job_descriptions = db.query(JobDescription).count()
    total_analyses = db.query(Analysis).count()
    
    # Verdict distribution
    verdicts = db.query(Analysis.verdict).all()
    verdict_counts = {"High": 0, "Medium": 0, "Low": 0}
    for (verdict,) in verdicts:
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1
    
    # Average scores
    scores = db.query(Analysis.overall_score).all()
    avg_score = sum(score[0] for score in scores) / len(scores) if scores else 0
    
    return {
        "total_resumes": total_resumes,
        "total_job_descriptions": total_job_descriptions,
        "total_analyses": total_analyses,
        "verdict_distribution": verdict_counts,
        "average_score": round(avg_score, 2),
        "processing_type": "enhanced" if hasattr(text_processor, 'calculate_enhanced_hard_score') else "basic"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)