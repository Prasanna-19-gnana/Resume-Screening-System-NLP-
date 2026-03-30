from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ParsedResumeSection(BaseModel):
    summary: str = ""
    skills: str = ""
    experience: str = ""
    projects: str = ""
    education: str = ""
    certifications: str = ""
    full_text: str = ""

class UploadResumeResponse(BaseModel):
    filename: str
    parsed_sections: ParsedResumeSection

class MatchRequest(BaseModel):
    parsed_resume: ParsedResumeSection = Field(..., description="The structured JSON returned by /upload-resume")
    job_role: str = Field(..., description="The title of the targeted job role")
    requirements: str = Field("", description="Comma-separated or bulleted list of mandatory requirements and skills")
    job_description: str = Field(..., description="Full text of the job description for semantic contextual matching")
    scorer_mode: str = Field("compare", description="Mode setting")

class MatchResponse(BaseModel):
    predicted_resume_role: str
    role_probabilities: Dict[str, float]
    role_alignment_score: float
    
    # Section-Aware Semantic Similarities
    skills_similarity: float
    projects_similarity: float
    experience_similarity: float
    semantic_similarity_score: float
    
    skill_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    
    final_score: float
    recommendation: str
