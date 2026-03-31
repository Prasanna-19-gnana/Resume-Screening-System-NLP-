"""
TEST SCRIPT: Validate All Fixes
Tests the complete pipeline with expected outcomes
"""

import sys
from pathlib import Path

# Add services to path
services_path = str(Path(__file__).parent / "app" / "services")
sys.path.insert(0, services_path)

from skill_extractor_fixed import SkillExtractor, extract_skills_from_text, match_skills
from semantic_matcher_fixed import SemanticMatcher
from role_detector_fixed import RoleDetector
from scorer_fixed import create_scorer


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_skill_extraction():
    """Test 1: Skill extraction works correctly"""
    print_section("TEST 1: Skill Extraction")
    
    skill_extractor = SkillExtractor()
    
    # Test case 1: Bracket handling
    text1 = "python (scikit-learn) machine learning (xgboost)"
    skills1 = extract_skills_from_text(text1)
    print(f"Input: {text1}")
    print(f"Extracted: {skills1}")
    assert "python" in skills1, "Failed: python not detected"
    assert "machine learning" in skills1, "Failed: machine learning not detected"
    assert "scikit-learn" in skills1, "Failed: scikit-learn not detected"
    print("✅ PASSED: Brackets handled correctly\n")
    
    # Test case 2: Common abbreviations
    text2 = "Python, ML, NLP, PyTorch"
    skills2 = extract_skills_from_text(text2)
    print(f"Input: {text2}")
    print(f"Extracted: {skills2}")
    assert "python" in skills2, "Failed: python abbrev not normalized"
    assert "machine learning" in skills2, "Failed: ml not normalized"
    assert "nlp" in skills2, "Failed: nlp not found"
    assert "pytorch" in skills2, "Failed: pytorch not found"
    print("✅ PASSED: Abbreviations normalized\n")
    
    # Test case 3: Multi-word skills
    text3 = "machine learning, deep learning, natural language processing"
    skills3 = skill_extractor.extract_from_requirements(text3)
    print(f"Input: {text3}")
    print(f"Extracted: {skills3}")
    assert "machine learning" in skills3, "Failed: machine learning not extracted"
    assert "nlp" in skills3, "Failed: nlp not normalized"
    print("✅ PASSED: Multi-word skills extracted\n")


def test_skill_matching():
    """Test 2: Skill matching logic"""
    print_section("TEST 2: Skill Matching")
    
    # Candidate has: Python, ML, Pandas, NumPy, XGBoost
    candidate = ["python", "machine learning", "pandas", "numpy", "xgboost"]
    
    # Required: Python, Machine Learning, Pandas, SQL, Tableau
    required = ["python", "machine learning", "pandas", "sql", "tableau"]
    
    matched, missing = match_skills(candidate, required)
    
    print(f"Candidate skills: {candidate}")
    print(f"Required skills: {required}")
    print(f"Matched: {matched}")
    print(f"Missing: {missing}")
    
    assert len(matched) == 3, f"Failed: Expected 3 matched, got {len(matched)}"
    assert set(missing) == {"sql", "tableau"}, f"Failed: Wrong missing skills"
    print("✅ PASSED: Skill matching correct\n")


def test_role_detection():
    """Test 3: Role detection with fallback"""
    print_section("TEST 3: Role Detection")
    
    detector = RoleDetector()
    
    # Test case 1: Data Scientist
    text1 = "Expert in machine learning, deep learning, TensorFlow, PyTorch, sklearn"
    role1 = detector.detect_role(text1)
    print(f"Text: ...{text1}...")
    print(f"Detected role: {role1}")
    assert "Data" in role1 or "Machine" in role1, "Failed: Should detect data/ML role"
    print("✅ PASSED: Data scientist detected\n")
    
    # Test case 2: Full Stack Developer
    text2 = "React, Node.js, Express, MongoDB, REST API"
    role2 = detector.detect_role(text2)
    print(f"Text: ...{text2}...")
    print(f"Detected role: {role2}")
    assert "Full" in role2 or "Frontend" in role2 or "Backend" in role2, "Failed: Should detect dev role"
    print("✅ PASSED: Developer role detected\n")
    
    # Test case 3: Role alignment scoring
    score1 = detector.get_role_alignment_score("Data Scientist", "Data Scientist")
    score2 = detector.get_role_alignment_score("Data Scientist", "Machine Learning Engineer")
    score3 = detector.get_role_alignment_score("Frontend Developer", "Backend Developer")
    
    print(f"Data Scientist vs Data Scientist: {score1} (expected 1.0)")
    print(f"Data Scientist vs ML Engineer: {score2} (expected 0.6-0.8)")
    print(f"Frontend vs Backend: {score3} (expected 0.6-0.8)")
    
    assert score1 == 1.0, "Failed: Exact match should be 1.0"
    assert 0.6 <= score2 <= 0.8, "Failed: Similar roles should score 0.6-0.8"
    print("✅ PASSED: Role alignment scores correct\n")


def test_end_to_end():
    """Test 4: End-to-end scoring"""
    print_section("TEST 4: End-to-End Scoring")
    
    # Initialize components
    skill_extractor = SkillExtractor()
    semantic_matcher = SemanticMatcher()
    role_detector = RoleDetector()
    scorer = create_scorer(skill_extractor, semantic_matcher, role_detector)
    
    # Mock resume
    resume_sections = {
        "skills": "Python, Machine Learning, Pandas, NumPy, XGBoost, TensorFlow, PyTorch",
        "experience": "5 years developing ML models, data analysis, statistical modeling",
        "projects": "Built recommendation system using KNN, implemented chatbot with NLP",
        "education": "BS Computer Science",
        "full_text": "Python, ML, Pandas, NumPy, XGBoost, TensorFlow, PyTorch, 5 years ML, built chatbot"
    }
    
    resume_semantic_text = " ".join([v for k, v in resume_sections.items() if k != "education"])
    
    # Job posting
    job_role = "Data Scientist"
    job_description = "Seeking experienced Data Scientist proficient in Python, ML, and deep learning"
    requirements = "Python, Machine Learning, TensorFlow, Pandas, Statistics"
    
    # Score
    result = scorer.score_candidate(
        parsed_resume=resume_sections,
        job_role=job_role,
        requirements=requirements,
        job_description=job_description,
        resume_semantic_text=resume_semantic_text
    )
    
    print(f"Job Role: {job_role}")
    print(f"Detected Role: {result['detected_role']}")
    print(f"\nScores:")
    print(f"  Final Score: {result['final_score']:.1f}/100")
    print(f"  Semantic: {result['semantic_similarity_score']:.1f}%")
    print(f"  Skills: {result['skill_match_score']:.1f}%")
    print(f"  Role Alignment: {result['role_alignment_score']:.1f}%")
    print(f"\nMatched Skills ({len(result['matched_skills'])}): {', '.join(result['matched_skills'][:5])}")
    print(f"Missing Skills ({len(result['missing_skills'])}): {', '.join(result['missing_skills'])}")
    print(f"\nRecommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']}")
    
    # Validate expectations
    assert result['final_score'] >= 60, f"Failed: Expected score >= 60, got {result['final_score']}"
    assert len(result['matched_skills']) >= 4, "Failed: Should match at least 4 skills"
    assert "Data" in result['detected_role'] or "Machine" in result['detected_role'], "Failed: Should detect data role"
    assert "STRONG" in result['recommendation'] or "GOOD" in result['recommendation'], "Failed: Should recommend"
    
    print("\n✅ PASSED: Complete scoring works correctly\n")


def test_consistency():
    """Test 5: No contradictions in output"""
    print_section("TEST 5: Output Consistency")
    
    skill_extractor = SkillExtractor()
    semantic_matcher = SemanticMatcher()
    role_detector = RoleDetector()
    scorer = create_scorer(skill_extractor, semantic_matcher, role_detector)
    
    # Edge case: All skills matched
    resume_sections = {
        "skills": "Python, Machine Learning, TensorFlow",
        "experience": "Senior ML engineer with 10 years experience",
        "projects": "Built advanced ML systems",
        "full_text": "Python, ML, TensorFlow, senior ML"
    }
    
    result = scorer.score_candidate(
        parsed_resume=resume_sections,
        job_role="Machine Learning Engineer",
        requirements="Python, Machine Learning, TensorFlow",
        job_description="Senior ML Engineer needed",
        resume_semantic_text=" ".join(resume_sections.values())
    )
    
    print(f"Edge case: All required skills present")
    print(f"  Final Score: {result['final_score']:.1f}/100")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Should NOT say "WEAK MATCH" if all skills matched
    assert "WEAK" not in result['recommendation'] and "POOR" not in result['recommendation'], \
        "Failed: Strong skill match should not result in weak recommendation"
    
    print("✅ PASSED: No contradictions\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  RESUME SCREENING SYSTEM - TEST SUITE")
    print("  Testing all fixes")
    print("="*60)
    
    try:
        test_skill_extraction()
        test_skill_matching()
        test_role_detection()
        test_end_to_end()
        test_consistency()
        
        print_section("✅ ALL TESTS PASSED")
        print("All fixes validated successfully!")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
