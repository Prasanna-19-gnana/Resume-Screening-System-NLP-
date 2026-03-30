from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from typing import Optional
from services.extraction.pdf_parser import extract_text_from_pdf
from services.extraction.docx_parser import extract_text_from_docx
from services.nlp.spacy_extractor import segment_sections
from services.nlp.skill_extractor import extract_skills, extract_skills_from_jd, extract_required_skills
from services.matching.semantic_scorer import compute_semantic_similarity, compute_experience_relevance, compute_education_relevance
from services.matching.weighted_ranker import rank_candidate
from services.explainability.score_explainer import generate_explanation

app = FastAPI(
    title="Resume Screening System (NLP)",
    description="E2E Pipeline for parsing, semantically matching, and scoring resumes against Job Descriptions.",
    version="1.3"
)

@app.get("/")
def health_check():
    return {"status": "success", "message": "Resume Screening API is active."}

@app.post("/api/v1/screen/")
async def screen_resume(
    job_role: str = Form(..., description="The Role title or category"),
    job_description: str = Form(..., description="The Job Description text for semantic context"),
    requirements: str = Form("", description="List of required skills or requirements"),
    scorer_mode: str = Form("compare", description="Mode: 'baseline', 'ml', or 'compare'"),
    resume: UploadFile = File(..., description="PDF or DOCX resume to be parsed")
):
    try:
        # Validate inputs
        if not job_role.strip():
            return {"error": "job_role is required."}
        if not job_description.strip():
            return {"error": "job_description is required."}

        # 1. Load File Data
        file_bytes = await resume.read()
        filename = resume.filename.lower()
        candidate_name = filename.rsplit('.', 1)[0].replace('_', ' ').title()
        
        # 2. Extract Base Text
        resume_text = ""
        if filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            resume_text = extract_text_from_docx(file_bytes)
        else:
            return {"error": "Unsupported file format. Upload PDF or DOCX."}
            
        if not resume_text:
            return {"error": "Could not extract text from document or document is empty."}
            
        # 3. NLP Segmentation
        sections = segment_sections(resume_text)
        
        # 4. Dictionary Skill Extraction
        candidate_skills = extract_skills(resume_text)
        
        warnings = []
        # Requirement Extraction with fallback
        if requirements.strip():
            raw_req_skills, normalized_req_skills = extract_required_skills(requirements)
        else:
            warnings.append("requirements_missing_using_fallback: true")
            normalized_req_skills = extract_skills_from_jd(job_description)
            raw_req_skills = normalized_req_skills
            
        # 5. Semantic AI Embeddings
        # Contextual matching using job_description
        overall_sim = compute_semantic_similarity(resume_text, job_description)
        combined_experience = sections.get("experience", "") + "\n" + sections.get("projects", "")
        exp_sim = compute_experience_relevance(combined_experience, job_description)
        
        # Calculate role alignment based on job_role instead of full JD
        role_alignment = compute_semantic_similarity(resume_text, job_role)
        
        # 6. Structuring Context for Logic
        parsed_resume = {
            "skills": candidate_skills,
            "experience_relevance": exp_sim,
            "overall_semantic_similarity": overall_sim,
            "role_alignment": role_alignment,
            "resume_text": resume_text  # Added for ML features
        }
        parsed_jd = {
            "job_role": job_role,
            "required_skills": normalized_req_skills
        }
        
        # 7. Computation & ML Inference
        scoring_result = rank_candidate(parsed_resume, parsed_jd, scorer_mode=scorer_mode)
        
        # 8. Explainability Engine
        # generate_explanation operates safely independently of the active scorer_mode
        explanation = generate_explanation(parsed_resume, parsed_jd, scoring_result)
        explanation["warnings"].extend(warnings)
        
        # --- DEBUGGING OUTPUT TO TERMINAL ---
        print("\n\n====== RESUME SCREENING DEBUG INFO ======")
        print(f"Candidate: {candidate_name}")
        print(f"Role: {job_role}")
        print(f"Required Skills Found: {normalized_req_skills}")
        print(f"Resume Skills Found: {candidate_skills}")
        if "baseline_score" in scoring_result:
            print(f"Baseline Score: {scoring_result['baseline_score']}")
        print("=========================================\n\n")

        # 9. Render Explicit Flat response Schema Requested
        # Always output the core identifiers & explainability
        response = {
            "job_role": job_role,
            "extracted_required_skills": raw_req_skills,
            "normalized_required_skills": normalized_req_skills,
            "extracted_resume_skills": candidate_skills,
            "matched_skills": explanation["matched_skills"],
            "missing_skills": explanation["missing_skills"],
            
            "role_alignment_score": scoring_result["raw_scores"].get("role_alignment", 0.0),
            "semantic_similarity_score": scoring_result["raw_scores"].get("semantic", 0.0),
            "final_score": scoring_result["final_score"],
            "fit_label": scoring_result.get("fit_label_prediction", "unknown"),
            
            "raw_scores": scoring_result.get("raw_scores", {}),
            "weighted_scores": scoring_result.get("weighted_scores", {}),
            
            "strengths": explanation["strengths"],
            "weak_areas": explanation["weak_areas"],
            "rejection_reasons": explanation.get("rejection_reasons", []),
            "warnings": explanation["warnings"]
        }
        
        # Optional: Include detailed scores conditionally based on requested mode
        if scorer_mode in ["baseline", "compare"]:
            response["baseline_score"] = scoring_result["baseline_score"]
            
        if scorer_mode in ["ml", "compare"]:
            response["ml_score"] = scoring_result.get("ml_score", 0.0)

        return response
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
