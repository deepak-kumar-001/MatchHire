# models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ResumeUpload(BaseModel):
    filename: str
    file_type: str

class JobDescriptionUpload(BaseModel):
    filename: str
    file_type: str

class AnalysisResult(BaseModel):
    id: int
    resume_filename: str
    hard_score: float
    soft_score: float
    overall_score: float
    verdict: str
    missing_skills: List[str]
    feedback: str
    analysis_date: datetime

class AnalysisRequest(BaseModel):
    resume_id: int
    job_description_id: int

class AnalysisResponse(BaseModel):
    hard_score: float
    soft_score: float
    overall_score: float
    verdict: str
    missing_skills: List[str]
    feedback: str
