from fastapi import FastAPI, HTTPException, UploadFile, File
import logging
import json

from .schemas import MatchRequest, MatchResponse, UploadResumeResponse, ParsedResumeSection
from .services.pdf_parser import extract_text_from_pdf, extract_text_from_docx
from .services.section_extractor import extract_sections
from .services.resume_builder import build_smart_resume_text
from .services.scorer import scorer_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production AI Resume Screening API",
    description="Advanced PDF Pipeline returning structurally embedded matches.",
    version="2.1.0"
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Advanced PDF Matching Service Active"}

@app.post("/upload-resume", response_model=UploadResumeResponse)
async def upload_resume(file: UploadFile = File(...)):
    """
    Robustly parses PDF/docx, segments via heuristics, and outputs structured analytical text blocks.
    """
    bytes_data = await file.read()
    filename = file.filename.lower()
    text = ""
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(bytes_data)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(bytes_data)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    logger.info(f"Extracted Raw Length: {len(text)} characters.")
    sections = extract_sections(text)
    
    parsed = ParsedResumeSection(
        summary=sections.get("summary", ""),
        skills=sections.get("skills", ""),
        experience=sections.get("experience", ""),
        projects=sections.get("projects", ""),
        education=sections.get("education", ""),
        certifications=sections.get("certifications", ""),
        full_text=text
    )
    return UploadResumeResponse(filename=filename, parsed_sections=parsed)

@app.post("/match", response_model=MatchResponse)
async def screen_candidate(req: MatchRequest):
    try:
        parsed_dict = req.parsed_resume.model_dump()
        smart_text = build_smart_resume_text(parsed_dict)
        
        # Scorer now fundamentally executes piece-wise embedding
        return scorer_service.score_candidate(
            parsed_resume_dict=parsed_dict,
            job_role=req.job_role,
            requirements=req.requirements,
            job_description=req.job_description,
            resume_semantic_text=smart_text
        )
    except Exception as e:
        logger.error(f"Screening Pipeline Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Ranking Error Occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
