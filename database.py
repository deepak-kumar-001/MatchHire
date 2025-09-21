# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./resume_analysis.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String)  # pdf, docx
    file_content = Column(LargeBinary)  # Store file binary data
    extracted_text = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)

class JobDescription(Base):
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String)
    file_content = Column(LargeBinary)
    extracted_text = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, index=True)
    job_description_id = Column(Integer, index=True)
    hard_score = Column(Float)
    soft_score = Column(Float)
    overall_score = Column(Float)
    verdict = Column(String)  # High, Medium, Low
    missing_skills = Column(Text)  # JSON string
    feedback = Column(Text)
    analysis_date = Column(DateTime, default=datetime.utcnow)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String)  # upload, analyze, view
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()