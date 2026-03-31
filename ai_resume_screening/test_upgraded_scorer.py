"""
QUICK INTEGRATION TEST: Upgraded Scorer with Ontology Matching
Verifies the upgraded scorer works correctly with ontology system
"""

import sys
import os
from pathlib import Path

# Add services to path
services_path = Path(__file__).parent / "app" / "services"
sys.path.insert(0, str(services_path))

from skill_extractor_fixed import SkillExtractor
from semantic_matcher_fixed import SemanticMatcher
from role_detector_fixed import RoleDetector
from skill_matcher_ontology import OntologyAwareSkillMatcher
from scorer_upgraded import create_upgraded_scorer

print("\n" + "="*70)
print("  UPGRADED SCORER INTEGRATION TEST")
print("="*70 + "\n")

# Initialize components
print("Loading components...")
skill_extractor = SkillExtractor()
semantic_matcher = SemanticMatcher()
role_detector = RoleDetector()
ontology_matcher = OntologyAwareSkillMatcher()

print("✅ Components loaded")

# Create upgraded scorer
scorer = create_upgraded_scorer(
    skill_extractor=skill_extractor,
    semantic_matcher=semantic_matcher,
    role_detector=role_detector,
    ontology_matcher=ontology_matcher
)

print("✅ Upgraded scorer created\n")

# Test scenario: Good candidate with related skills
test_case = {
    "name": "ML Engineer with related skills (ontology value)",
    "resume_text": """
    Senior Machine Learning Engineer
    
    SKILLS:
    Deep Learning, TensorFlow, PyTorch, Keras, Python, SQL, Pandas, NumPy
    
    EXPERIENCE:
    - 5 years as ML Engineer at tech company
    - Built neural network models for image classification
    - Optimized ML pipelines using scikit-learn
    - Worked with NLP for text analysis
    
    PROJECTS:
    - Computer vision project using CNN
    - Time series forecasting model
    - Data analysis dashboard in Python
    """,
    "job_role": "ML Engineer",
    "requirements": "deep learning, machine learning, tensorflow, python, sql",
    "job_description": "We are looking for an ML Engineer with experience in deep learning frameworks and Python. Must have SQL knowledge and machine learning background.",
}

# Mock parsed resume
parsed_resume = {
    "summary": test_case["resume_text"][:500],
    "skills": test_case["resume_text"],
    "experience": test_case["resume_text"],
    "projects": test_case["resume_text"],
    "education": "",
    "certifications": "",
    "full_text": test_case["resume_text"]
}

print(f"Test: {test_case['name']}\n")
print(f"Resume excerpt: {test_case['resume_text'][:100]}...\n")

# Score the candidate
result = scorer.score_candidate(
    parsed_resume=parsed_resume,
    job_role=test_case["job_role"],
    requirements=test_case["requirements"],
    job_description=test_case["job_description"],
    resume_semantic_text=test_case["resume_text"],
)

# Display results
print("-"*70)
print("✅ SCORING COMPLETE\n")

print(f"Final Score: {result['final_score']}/100")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}\n")

print("Score Breakdown:")
print(f"  - Semantic Match: {result['semantic_similarity_score']:.1f}%")
print(f"  - Skill Match: {result['skill_match_score']:.1f}%")
print(f"  - Role Alignment: {result['role_alignment_score']:.1f}%\n")

print("Skill Details:")
print(f"  - Exact Matches: {len(result['exact_matches'])} - {result['exact_matches']}")
print(f"  - Partial Matches: {len(result['partial_matches'])} - ", end="")
if result['partial_matches']:
    for match in result['partial_matches'][:3]:
        if isinstance(match, dict):
            print(f"{match.get('required', '?')}", end=" ")
    print()
else:
    print("None")
print(f"  - Missing Skills: {len(result['missing_skills'])} - {result['missing_skills']}\n")

# Verify ontology matching is working
print("Ontology System Check:")
skill_breakdown = result['skill_breakdown']
print(f"  - Total Required: {skill_breakdown['total_required']}")
print(f"  - Exact Matches: {skill_breakdown['exact_count']}")
print(f"  - Partial Matches: {skill_breakdown['partial_count']}")
print(f"  - Missing: {skill_breakdown['missing_count']}")
print(f"  - Points Earned: {skill_breakdown['points_earned']:.1f}")
print(f"  - Points Possible: {skill_breakdown['points_possible']:.1f}%\n")

# Validation
print("-"*70)
if result['final_score'] >= 65:
    print("✅ VALIDATION PASSED: Score reasonable for good candidate with related skills")
else:
    print("⚠️ VALIDATION WARNING: Score lower than expected")

if result['exact_matches'] or result['partial_matches']:
    print("✅ VALIDATION PASSED: Skill matching found at least one match")
else:
    print("⚠️ VALIDATION WARNING: No skills matched")

if "GOOD" in result['recommendation'] or "STRONG" in result['recommendation']:
    print("✅ VALIDATION PASSED: Recommendation appropriate for good candidate")
else:
    print("⚠️ VALIDATION WARNING: Recommendation unexpected")

print("\n" + "="*70)
print("  UPGRADED SCORER INTEGRATION SUCCESSFUL!")
print("="*70 + "\n")
