"""
COMPREHENSIVE DEBUG SCRIPT: Full Pipeline Visibility
Traces every stage of the resume screening pipeline to identify where scoring breaks

USAGE:
    python debug_pipeline.py [--verbose]
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "ai_resume_screening" / "app"))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# PART 1: INPUT VALIDATION
# ============================================================

def validate_inputs(job_role, job_description):
    """Validate inputs at entry point"""
    
    print("\n" + "="*70)
    print("PART 1: INPUT VALIDATION")
    print("="*70)
    
    # Check job role
    print(f"\n[INPUT] Job Role: '{job_role}'")
    print(f"[LENGTH] {len(job_role)} characters")
    print(f"[WORDS] {len(job_role.split())} words")
    
    if not job_role or len(job_role.strip()) == 0:
        raise ValueError("❌ Job role is empty!")
    
    if len(job_role.split()) > 6:
        print(f"⚠️  WARNING: Job role appears too long ({len(job_role.split())} words)")
        print(f"   Expected: 'Data Scientist', Got: '{job_role}'")
    
    # Check job description
    print(f"\n[INPUT] Job Description Length: {len(job_description)} characters")
    
    if len(job_description) < 30:
        raise ValueError(f"❌ Job description too short ({len(job_description)} chars)")
    
    print(f"[PREVIEW] {job_description[:200]}...")
    print("\n✅ Input validation PASSED")
    
    return True


# ============================================================
# PART 2: PDF PARSING DEBUG
# ============================================================

def debug_pdf_parsing(resume_text):
    """Debug PDF/resume text extraction"""
    
    print("\n" + "="*70)
    print("PART 2: PDF PARSING DEBUG")
    print("="*70)
    
    print(f"\n[EXTRACTED] Total characters: {len(resume_text)}")
    print(f"[EXTRACTED] Total words: {len(resume_text.split())}")
    
    if len(resume_text) == 0:
        raise ValueError("❌ Resume text is empty!")
    
    if len(resume_text) < 50:
        print(f"⚠️  WARNING: Resume text very short ({len(resume_text)} chars)")
    
    print(f"\n[RAW TEXT SAMPLE (first 500 chars)]:")
    print("-" * 70)
    print(resume_text[:500])
    print("-" * 70)
    
    # Check for common issues
    has_skills = 'skill' in resume_text.lower()
    has_experience = 'experience' in resume_text.lower() or 'work' in resume_text.lower()
    has_projects = 'project' in resume_text.lower()
    
    print(f"\n[CONTENT CHECK]")
    print(f"  Contains 'skill': {has_skills}")
    print(f"  Contains 'experience'/'work': {has_experience}")
    print(f"  Contains 'project': {has_projects}")
    
    if not any([has_skills, has_experience, has_projects]):
        print("⚠️  WARNING: Resume appears to contain no standard sections")
    
    print("\n✅ PDF parsing DEBUG complete")
    return resume_text


# ============================================================
# PART 3: SECTION EXTRACTION DEBUG
# ============================================================

def debug_section_extraction(sections):
    """Debug section extraction"""
    
    print("\n" + "="*70)
    print("PART 3: SECTION EXTRACTION DEBUG")
    print("="*70)
    
    section_keys = list(sections.keys())
    print(f"\n[SECTIONS FOUND]: {section_keys}")
    
    for section_name, section_text in sections.items():
        length = len(section_text) if section_text else 0
        words = len(section_text.split()) if section_text else 0
        
        print(f"\n[{section_name.upper()}]")
        print(f"  Length: {length} characters")
        print(f"  Words: {words}")
        
        if length > 0:
            print(f"  Preview: {section_text[:200]}...")
        else:
            print(f"  ⚠️  EMPTY SECTION")
    
    # Critical: Skills section
    if sections.get("skills"):
        print(f"\n✅ Skills section present ({len(sections['skills'])} chars)")
    else:
        print(f"\n⚠️  Skills section is empty or missing")
    
    print("\n✅ Section extraction DEBUG complete")
    return sections


# ============================================================
# PART 4: SKILL EXTRACTION DEBUG
# ============================================================

def debug_skill_extraction(skill_extractor, resume_text, job_description, requirements_text):
    """Debug skill extraction"""
    
    print("\n" + "="*70)
    print("PART 4: SKILL EXTRACTION DEBUG")
    print("="*70)
    
    # Extract from resume
    print(f"\n[EXTRACTING FROM RESUME]")
    print(f"  Input length: {len(resume_text)} chars")
    
    try:
        resume_skills = skill_extractor.extract_skills(resume_text)
        print(f"  ✅ Extracted {len(resume_skills)} skills from resume")
        print(f"  Skills: {resume_skills[:15]}")  # First 15
        
        if not resume_skills:
            print(f"  ⚠️  WARNING: No skills extracted from resume!")
    except Exception as e:
        print(f"  ❌ ERROR extracting resume skills: {e}")
        resume_skills = []
    
    # Extract from JD
    print(f"\n[EXTRACTING FROM JOB DESCRIPTION]")
    print(f"  Input length: {len(job_description)} chars")
    
    try:
        jd_skills_from_desc = skill_extractor.extract_skills(job_description)
        print(f"  ✅ Extracted {len(jd_skills_from_desc)} skills from JD description")
        print(f"  Skills: {jd_skills_from_desc[:15]}")
    except Exception as e:
        print(f"  ❌ ERROR extracting JD description skills: {e}")
        jd_skills_from_desc = []
    
    # Extract from requirements
    if requirements_text and len(requirements_text.strip()) > 0:
        print(f"\n[EXTRACTING FROM REQUIREMENTS]")
        print(f"  Input length: {len(requirements_text)} chars")
        print(f"  Input: {requirements_text[:200]}...")
        
        try:
            jd_skills_from_req = skill_extractor.extract_from_requirements(requirements_text)
            print(f"  ✅ Extracted {len(jd_skills_from_req)} skills from requirements")
            print(f"  Skills: {jd_skills_from_req[:15]}")
        except Exception as e:
            print(f"  ❌ ERROR extracting requirements skills: {e}")
            jd_skills_from_req = []
    else:
        print(f"\n[REQUIREMENTS] Empty or Not Provided")
        jd_skills_from_req = []
    
    # Combine JD skills
    jd_skills = list(set(jd_skills_from_desc + jd_skills_from_req))
    print(f"\n[COMBINED JD SKILLS]")
    print(f"  Total unique: {len(jd_skills)}")
    print(f"  Skills: {jd_skills[:15]}")
    
    # Critical checks
    if not resume_skills:
        print(f"\n⚠️  CRITICAL: No skills extracted from resume")
    
    if not jd_skills:
        print(f"\n⚠️  CRITICAL: No skills extracted from JD")
    
    print("\n✅ Skill extraction DEBUG complete")
    return resume_skills, jd_skills


# ============================================================
# PART 5: SKILL MATCHING DEBUG
# ============================================================

def debug_skill_matching(resume_skills, jd_skills):
    """Debug skill matching logic"""
    
    print("\n" + "="*70)
    print("PART 5: SKILL MATCHING DEBUG")
    print("="*70)
    
    print(f"\n[INPUTS]")
    print(f"  Resume skills: {len(resume_skills)} total")
    print(f"  JD skills: {len(jd_skills)} total")
    
    if len(resume_skills) == 0 or len(jd_skills) == 0:
        print(f"\n❌ CRITICAL: Cannot match skills - one or both lists empty")
        print(f"   This will result in 0.0 skill match score!")
        return 0.0, [], [], []
    
    # Exact matches (case-insensitive)
    exact_matches = []
    for req_skill in jd_skills:
        for res_skill in resume_skills:
            if req_skill.lower() == res_skill.lower():
                exact_matches.append(req_skill)
                break
    
    print(f"\n[EXACT MATCHES]")
    print(f"  Count: {len(exact_matches)}")
    print(f"  Skills: {exact_matches}")
    
    # Partial matches
    partial_matches = []
    for req_skill in jd_skills:
        if req_skill.lower() not in [s.lower() for s in resume_skills]:
            for res_skill in resume_skills:
                if req_skill.lower() in res_skill.lower() or res_skill.lower() in req_skill.lower():
                    partial_matches.append(req_skill)
                    break
    
    print(f"\n[PARTIAL MATCHES]")
    print(f"  Count: {len(partial_matches)}")
    print(f"  Skills: {partial_matches}")
    
    # Missing skills
    missing_skills = [s for s in jd_skills if s.lower() not in [r.lower() for r in resume_skills]]
    
    print(f"\n[MISSING SKILLS]")
    print(f"  Count: {len(missing_skills)}")
    print(f"  Skills: {missing_skills}")
    
    # Compute skill match score
    total_matched = len(exact_matches) + (len(partial_matches) * 0.5)
    skill_score = total_matched / len(jd_skills) if jd_skills else 0.0
    
    print(f"\n[SKILL SCORE CALCULATION]")
    print(f"  Exact: {len(exact_matches)} × 1.0 = {len(exact_matches)}")
    print(f"  Partial: {len(partial_matches)} × 0.5 = {len(partial_matches) * 0.5}")
    print(f"  Total matched: {total_matched}")
    print(f"  JD skills: {len(jd_skills)}")
    print(f"  Score: {total_matched} / {len(jd_skills)} = {skill_score:.4f}")
    
    if skill_score == 0.0:
        print(f"\n⚠️  CRITICAL: Skill match score is 0.0!")
    
    print("\n✅ Skill matching DEBUG complete")
    return skill_score, exact_matches, partial_matches, missing_skills


# ============================================================
# PART 6: SEMANTIC MATCHING DEBUG
# ============================================================

def debug_semantic_matching(semantic_matcher, resume_text, job_description):
    """Debug semantic similarity computation"""
    
    print("\n" + "="*70)
    print("PART 6: SEMANTIC MATCHING DEBUG")
    print("="*70)
    
    print(f"\n[INPUTS]")
    print(f"  Resume text length: {len(resume_text)} chars")
    print(f"  JD text length: {len(job_description)} chars")
    print(f"  Resume preview: {resume_text[:100]}...")
    print(f"  JD preview: {job_description[:100]}...")
    
    try:
        # Attempt similarity computation
        similarity = semantic_matcher.compute_similarity(resume_text[:500], job_description[:500])
        
        print(f"\n[SEMANTIC SIMILARITY]")
        print(f"  Raw value: {similarity}")
        print(f"  Type: {type(similarity)}")
        
        if similarity == 0.0:
            print(f"  ⚠️  WARNING: Semantic similarity is 0.0")
            print(f"     Possible causes:")
            print(f"       - Empty resume or JD text")
            print(f"       - Embeddings model not loaded")
            print(f"       - Mismatch in text format")
        elif similarity < 0 or similarity > 1:
            print(f"  ⚠️  WARNING: Semantic similarity out of range [0, 1]")
        
        print(f"\n✅ Semantic value: {similarity:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR computing semantic similarity: {e}")
        print(f"   This will result in 0.0 semantic score")
        similarity = 0.0
    
    print("\n✅ Semantic matching DEBUG complete")
    return similarity


# ============================================================
# PART 7: ROLE DETECTION DEBUG
# ============================================================

def debug_role_detection(role_detector, resume_text, job_role):
    """Debug role detection"""
    
    print("\n" + "="*70)
    print("PART 7: ROLE DETECTION DEBUG")
    print("="*70)
    
    print(f"\n[TARGET JOB ROLE]")
    print(f"  Role: '{job_role}'")
    print(f"  Expected type: String like 'Data Scientist'")
    
    if not job_role or len(job_role.strip()) == 0:
        print(f"  ⚠️  WARNING: Job role is empty!")
    
    print(f"\n[RESUME ROLE DETECTION]")
    print(f"  Input: {resume_text[:200]}...")
    
    try:
        detected_role = role_detector.detect_role(resume_text)
        print(f"  Detected: '{detected_role}'")
        print(f"  Type: {type(detected_role)}")
        
        if not detected_role or detected_role == "Unknown":
            print(f"  ⚠️  WARNING: Could not detect role from resume")
            detected_role = "Unknown"
        
    except Exception as e:
        print(f"  ❌ ERROR detecting role: {e}")
        detected_role = "Unknown"
    
    # Role alignment
    print(f"\n[ROLE ALIGNMENT]")
    print(f"  Target: '{job_role}'")
    print(f"  Detected: '{detected_role}'")
    
    try:
        role_score = role_detector.get_role_alignment_score(detected_role, job_role)
        print(f"  Alignment score: {role_score:.4f}")
        
        if role_score == 0.0:
            print(f"  ⚠️  WARNING: Role alignment is 0.0")
        
    except Exception as e:
        print(f"  ❌ ERROR computing role alignment: {e}")
        role_score = 0.5  # Default
    
    print("\n✅ Role detection DEBUG complete")
    return detected_role, role_score


# ============================================================
# PART 8: FEATURE VECTOR DEBUG
# ============================================================

def debug_feature_vector(semantic_score, skill_score, role_score):
    """Debug feature vector construction"""
    
    print("\n" + "="*70)
    print("PART 8: FEATURE VECTOR DEBUG")
    print("="*70)
    
    print(f"\n[RAW SCORES]")
    print(f"  Semantic: {semantic_score:.4f}")
    print(f"  Skill: {skill_score:.4f}")
    print(f"  Role: {role_score:.4f}")
    
    # Check for all zeros
    if semantic_score == 0 and skill_score == 0 and role_score == 0:
        print(f"\n⚠️  CRITICAL: ALL FEATURES ARE ZERO")
        print(f"     This will result in 0.0 final score!")
        print(f"     Check Parts 4, 5, 6 for failure sources")
    
    # Weighted combination
    weights = {"semantic": 0.4, "skill": 0.4, "role": 0.2}
    weighted_sum = (semantic_score * weights["semantic"] + 
                   skill_score * weights["skill"] + 
                   role_score * weights["role"])
    
    print(f"\n[WEIGHTED COMBINATION]")
    print(f"  Semantic: {semantic_score:.4f} × {weights['semantic']} = {semantic_score * weights['semantic']:.4f}")
    print(f"  Skill: {skill_score:.4f} × {weights['skill']} = {skill_score * weights['skill']:.4f}")
    print(f"  Role: {role_score:.4f} × {weights['role']} = {role_score * weights['role']:.4f}")
    print(f"  Total: {weighted_sum:.4f}")
    
    # Sanity check
    if weighted_sum < 0 or weighted_sum > 1:
        print(f"\n⚠️  WARNING: Weighted sum out of range [0, 1]")
    
    print("\n✅ Feature vector DEBUG complete")
    return weighted_sum


# ============================================================
# PART 9: MODEL PREDICTION DEBUG
# ============================================================

def debug_model_prediction(ml_scorer, feature_dict):
    """Debug ML model prediction"""
    
    print("\n" + "="*70)
    print("PART 9: MODEL PREDICTION DEBUG")
    print("="*70)
    
    print(f"\n[ML MODEL CHECK] ")
    
    try:
        # Try to score
        result = ml_scorer.score_resume(
            resume_text="Test",
            job_description="Test",
            requirements="Test"
        )
        
        predicted_score = result.get("score", 0.0)
        
        print(f"  Model loaded: YES")
        print(f"  Predicted score (0-100): {predicted_score:.2f}")
        print(f"  Prediction method: {result.get('method', 'unknown')}")
        
        if predicted_score == 0.0:
            print(f"  ⚠️  WARNING: Model predicted 0.0 score")
        
    except Exception as e:
        print(f"  ❌ ERROR with ML model: {e}")
        print(f"     Will fall back to rule-based scoring")
        predicted_score = None
    
    print("\n✅ Model prediction DEBUG complete")
    return predicted_score


# ============================================================
# PART 10: SANITY CHECK
# ============================================================

def sanity_check(semantic_score, skill_score, role_score, final_score):
    """Perform sanity checks"""
    
    print("\n" + "="*70)
    print("PART 10: SANITY CHECK")
    print("="*70)
    
    issues = []
    
    print(f"\n[CHECK 1] Feature ranges")
    if not (0 <= semantic_score <= 1):
        issues.append(f"Semantic score out of range: {semantic_score}")
    if not (0 <= skill_score <= 1):
        issues.append(f"Skill score out of range: {skill_score}")
    if not (0 <= role_score <= 1):
        issues.append(f"Role score out of range: {role_score}")
    
    print(f"  Semantic [0-1]: {semantic_score:.4f} ✅" if 0 <= semantic_score <= 1 else f"  Semantic [0-1]: {semantic_score:.4f} ❌")
    print(f"  Skill [0-1]: {skill_score:.4f} ✅" if 0 <= skill_score <= 1 else f"  Skill [0-1]: {skill_score:.4f} ❌")
    print(f"  Role [0-1]: {role_score:.4f} ✅" if 0 <= role_score <= 1 else f"  Role [0-1]: {role_score:.4f} ❌")
    
    print(f"\n[CHECK 2] Final score")
    if not (0 <= final_score <= 100):
        issues.append(f"Final score out of range: {final_score}")
    
    print(f"  Final [0-100]: {final_score:.2f} ✅" if 0 <= final_score <= 100 else f"  Final [0-100]: {final_score:.2f} ❌")
    
    print(f"\n[CHECK 3] All zeros?")
    if semantic_score == 0 and skill_score == 0 and role_score == 0:
        issues.append("All features are zero - pipeline failed")
        print(f"  All zeros: YES ❌")
    else:
        print(f"  All zeros: NO ✅")
    
    print(f"\n[CHECK 4] At least one non-zero?")
    if max(semantic_score, skill_score, role_score) > 0:
        print(f"  At least one non-zero: YES ✅")
    else:
        issues.append("No non-zero features")
        print(f"  At least one non-zero: NO ❌")
    
    # Report issues
    if issues:
        print(f"\n⚠️  ISSUES FOUND: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ All sanity checks passed!")
    
    return len(issues) == 0


# ============================================================
# PART 11: FALLBACK SCORING
# ============================================================

def compute_fallback_score(semantic_score, skill_score, role_score):
    """Compute fallback score if model fails"""
    
    print("\n" + "="*70)
    print("PART 11: FALLBACK SCORING")
    print("="*70)
    
    # Simple weighted combination
    fallback_score = (0.4 * semantic_score + 
                     0.4 * skill_score + 
                     0.2 * role_score)
    
    print(f"\n[FALLBACK FORMULA]")
    print(f"  Score = (Semantic × 0.4) + (Skill × 0.4) + (Role × 0.2)")
    print(f"  Score = ({semantic_score:.4f} × 0.4) + ({skill_score:.4f} × 0.4) + ({role_score:.4f} × 0.2)")
    print(f"  Score = {semantic_score * 0.4:.4f} + {skill_score * 0.4:.4f} + {role_score * 0.2:.4f}")
    print(f"  Score = {fallback_score:.4f}")
    
    # Convert to 0-100
    fallback_score_100 = fallback_score * 100
    
    print(f"\n[CONVERTED TO 0-100 SCALE]")
    print(f"  Score: {fallback_score_100:.2f}/100")
    
    return fallback_score_100


# ============================================================
# MAIN TEST CASE
# ============================================================

def run_debug_pipeline():
    """Run complete debug pipeline"""
    
    print("\n" + "="*70)
    print("FULL PIPELINE DEBUG TRACE")
    print("="*70)
    print(f"Object ID: {id(run_debug_pipeline)}")
    print(f"Timestamp: {Path.cwd()}")
    
    # TEST CASE: Data Scientist
    job_role = "Data Scientist"
    job_description = """
    We are looking for an experienced Data Scientist to join our team.
    
    Key responsibilities:
    - Build and maintain machine learning models
    - Analyze large datasets and derive insights
    - Design and implement predictive models
    - Collaborate with product and engineering teams
    
    Required skills:
    - Python programming (5+ years)
    - Machine learning frameworks (scikit-learn, TensorFlow, PyTorch)
    - Statistical modeling and data analysis
    - SQL and big data processing
    - Deep learning experience
    """
    
    requirements = "Python, Machine Learning, TensorFlow, PyTorch, SQL, Data Analysis, Statistics"
    
    # Create sample resume
    sample_resume = """
    John Data
    john@email.com | linkedin.com/in/johndata
    
    SUMMARY
    Senior Data Scientist with 6 years of experience building predictive models
    and analyzing complex datasets.
    
    SKILLS
    - Python: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
    - Machine Learning: Classification, Regression, NLP, Computer Vision
    - Data Analysis: SQL, Tableau, Excel
    - Big Data: Spark, Hadoop
    - Statistical Modeling: A/B Testing, Hypothesis Testing
    
    EXPERIENCE
    Senior Data Scientist | Tech Corp | 2021-Present
    - Developed ML models improving prediction accuracy by 35%
    - Led data pipeline architecture serving 1M+ predictions/day
    - Mentored junior data scientists
    
    Data Scientist | Analytics Inc | 2019-2021
    - Built deep learning models for NLP tasks
    - Conducted statistical analysis for business decisions
    
    EDUCATION
    M.S. Computer Science | State University | 2019
    B.S. Mathematics | State University | 2017
    """
    
    try:
        # PART 1: Input Validation
        validate_inputs(job_role, job_description)
        
        # PART 2: PDF Parsing
        resume_text = debug_pdf_parsing(sample_resume)
        
        # PART 3: Section Extraction
        from services.section_extractor import extract_sections
        sections = extract_sections(resume_text)
        debug_section_extraction(sections)
        
        # PART 4: Skill Extraction
        from services.skill_extractor_fixed import SkillExtractor
        skill_extractor = SkillExtractor()
        resume_skills, jd_skills = debug_skill_extraction(
            skill_extractor, resume_text, job_description, requirements
        )
        
        # PART 5: Skill Matching
        skill_score, exact_matches, partial_matches, missing_skills = debug_skill_matching(
            resume_skills, jd_skills
        )
        
        # PART 6: Semantic Matching
        from services.semantic_matcher_fixed import SemanticMatcher
        semantic_matcher = SemanticMatcher()
        semantic_score = debug_semantic_matching(semantic_matcher, resume_text, job_description)
        
        # PART 7: Role Detection
        from services.role_detector_fixed import RoleDetector
        role_detector = RoleDetector()
        detected_role, role_score = debug_role_detection(role_detector, resume_text, job_role)
        
        # PART 8: Feature Vector
        rule_based_score = debug_feature_vector(semantic_score, skill_score, role_score)
        rule_based_score_100 = rule_based_score * 100
        
        # PART 9: Model Prediction
        from services.ml_scorer_service import ml_scorer_service
        ml_predicted = debug_model_prediction(ml_scorer_service, {
            "semantic": semantic_score,
            "skill": skill_score,
            "role": role_score
        })
        
        # PART 10: Sanity Check
        sanity_ok = sanity_check(semantic_score, skill_score, role_score, rule_based_score_100)
        
        # PART 11: Fallback Scoring
        fallback_score = compute_fallback_score(semantic_score, skill_score, role_score)
        
        # FINAL SUMMARY
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print(f"\n[SCORES]")
        print(f"  Semantic Similarity: {semantic_score:.4f} (scaled: {semantic_score*100:.2f}%)")
        print(f"  Skill Match: {skill_score:.4f} (scaled: {skill_score*100:.2f}%)")
        print(f"  Role Alignment: {role_score:.4f} (scaled: {role_score*100:.2f}%)")
        print(f"\n[FINAL SCORES]")
        print(f"  Rule-Based: {rule_based_score_100:.2f}/100")
        print(f"  Fallback: {fallback_score:.2f}/100")
        
        if ml_predicted is not None:
            print(f"  ML Model: {ml_predicted:.2f}/100")
        else:
            print(f"  ML Model: NOT AVAILABLE")
        
        print(f"\n[SKILLS]")
        print(f"  Exact Matches: {len(exact_matches)} - {exact_matches}")
        print(f"  Partial Matches: {len(partial_matches)} - {partial_matches}")
        print(f"  Missing: {len(missing_skills)} - {missing_skills}")
        
        print(f"\n[ROLE]")
        print(f"  Detected: {detected_role}")
        print(f"  Target: {job_role}")
        
        print(f"\n[SANITY CHECK]")
        print(f"  Status: {'✅ PASSED' if sanity_ok else '❌ FAILED'}")
        
        if rule_based_score_100 == 0:
            print(f"\n⚠️  CRITICAL ISSUE: Final score is 0")
            print(f"    Review the sections above to identify failure point:")
            if semantic_score == 0:
                print(f"    ❌ Semantic matching failed (see PART 6)")
            if skill_score == 0:
                print(f"    ❌ Skill matching failed (see PART 5)")
            if role_score == 0:
                print(f"    ❌ Role detection failed (see PART 7)")
        else:
            print(f"\n✅ Pipeline producing realistic scores:")
            print(f"    Score: {rule_based_score_100:.2f}/100")
            print(f"    Recommendation: {'STRONG' if rule_based_score_100 >= 80 else 'GOOD' if rule_based_score_100 >= 65 else 'MODERATE' if rule_based_score_100 >= 45 else 'WEAK'}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_debug_pipeline()
