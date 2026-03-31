"""
COMPREHENSIVE TEST SUITE: Ontology-Aware Skill Matching
- Tests for skill ontology
- Tests for partial matching logic
- Tests for upgraded scorer
- Tests for realistic scenarios
"""

import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app", "services"))

from skill_ontology import SkillOntology
from skill_matcher_ontology import OntologyAwareSkillMatcher
from skill_extractor_fixed import SkillExtractor
from semantic_matcher_fixed import SemanticMatcher
from role_detector_fixed import RoleDetector
from scorer_upgraded import create_upgraded_scorer

print("\n" + "="*70)
print("  COMPREHENSIVE ONTOLOGY TEST SUITE")
print("="*70 + "\n")

# ============================================================
# PART 1: SKILL ONTOLOGY TESTS
# ============================================================

print("PART 1: SKILL ONTOLOGY VALIDATION")
print("-" * 70)

ontology = SkillOntology()

test_cases_ontology = [
    {
        "name": "Check machine learning hierarchy",
        "skill": "scikit-learn",
        "expected": ["machine learning"],
        "test": lambda: ontology.get_parents_of_skill("scikit-learn")
    },
    {
        "name": "Check deep learning children",
        "skill": "deep learning",
        "expected": ["tensorflow", "pytorch", "keras", "neural networks"],
        "test": lambda: ontology.get_children_of_skill("deep learning"),
        "check": "contains"  # At least these should be present
    },
    {
        "name": "Check NLP hierarchy",
        "skill": "transformers",
        "expected": ["nlp"],
        "test": lambda: ontology.get_parents_of_skill("transformers")
    },
    {
        "name": "Check data tools parents",
        "skill": "pandas",
        "expected": ["data analysis"],
        "test": lambda: ontology.get_parents_of_skill("pandas")
    },
    {
        "name": "Verify parent skill",
        "skill": "machine learning",
        "expected": True,
        "test": lambda: ontology.is_parent_skill("machine learning")
    },
]

ontology_passed = 0
for test in test_cases_ontology:
    result = test["test"]()
    
    if test.get("check") == "contains":
        # Check if expected items are contained in result
        success = all(item in result for item in test["expected"])
    else:
        success = result == test["expected"]
    
    status = "✅" if success else "❌"
    print(f"{status} {test['name']}")
    print(f"   Expected: {test['expected']}")
    print(f"   Got: {result}")
    
    if success:
        ontology_passed += 1

print(f"\nOntology Tests: {ontology_passed}/{len(test_cases_ontology)} PASSED\n")

# ============================================================
# PART 2: ONTOLOGY-AWARE MATCHER TESTS
# ============================================================

print("PART 2: ONTOLOGY-AWARE MATCHING TESTS")
print("-" * 70)

matcher = OntologyAwareSkillMatcher()

matcher_test_cases = [
    {
        "name": "Exact match: TensorFlow → TensorFlow",
        "candidate": ["tensorflow"],
        "required": ["tensorflow"],
        "expected_exact": 1,
        "expected_partial": 0,
        "expected_missing": 0,
        "min_score": 0.9,
    },
    {
        "name": "Partial match: Deep Learning → TensorFlow",
        "candidate": ["deep learning"],
        "required": ["tensorflow"],
        "expected_exact": 0,
        "expected_partial": 1,
        "expected_missing": 0,
        "min_score": 0.6,
    },
    {
        "name": "Partial match: PyTorch → Deep Learning",
        "candidate": ["pytorch"],
        "required": ["deep learning"],
        "expected_exact": 0,
        "expected_partial": 1,
        "expected_missing": 0,
        "min_score": 0.6,
    },
    {
        "name": "Mixed matching: Some exact, some partial, some missing",
        "candidate": ["tensorflow", "deep learning", "pandas", "python"],
        "required": ["tensorflow", "pytorch", "machine learning", "sql"],
        "expected_exact": 1,
        "expected_partial": 1,  # pytorch gets parent match (deep learning is parent of pytorch)
        "expected_missing": 2,  # machine learning and sql not matched (ML could be matched but depends on logic)
        "min_score": 0.3,  # Lower expectation since many misses
    },
    {
        "name": "No match: Different domains",
        "candidate": ["python", "javascript"],
        "required": ["tensorflow", "pytorch"],
        "expected_exact": 0,
        "expected_partial": 0,
        "expected_missing": 2,
        "max_score": 0.3,
    },
]

matcher_passed = 0
for test in matcher_test_cases:
    result = matcher.match_skills(test["candidate"], test["required"])
    
    exact_count = len(result["exact_matches"])
    partial_count = len(result["partial_matches"])
    missing_count = len(result["missing_skills"])
    score = result["skill_match_score"]
    
    # Validate structure
    structure_ok = all(key in result for key in [
        "exact_matches", "partial_matches", "missing_skills",
        "skill_match_score", "score_breakdown"
    ])
    
    # Check counts
    exact_ok = exact_count == test["expected_exact"]
    partial_ok = partial_count == test["expected_partial"]
    missing_ok = missing_count == test["expected_missing"]
    
    # Check score bounds
    score_ok = True
    if "min_score" in test:
        score_ok = score >= test["min_score"]
    if "max_score" in test:
        score_ok = score <= test["max_score"]
    
    success = structure_ok and exact_ok and partial_ok and missing_ok and score_ok
    
    status = "✅" if success else "❌"
    print(f"{status} {test['name']}")
    print(f"   Candidate: {test['candidate']}")
    print(f"   Required:  {test['required']}")
    print(f"   Exact: {exact_count} (expected {test['expected_exact']}) | "
          f"Partial: {partial_count} (expected {test['expected_partial']}) | "
          f"Missing: {missing_count} (expected {test['expected_missing']})")
    print(f"   Score: {score:.3f}" + 
          (f" (expected ≥{test.get('min_score', 0):.1f})" if "min_score" in test else "") +
          (f" (expected ≤{test.get('max_score', 1):.1f})" if "max_score" in test else ""))
    
    if success:
        matcher_passed += 1

print(f"\nMatcher Tests: {matcher_passed}/{len(matcher_test_cases)} PASSED\n")

# ============================================================
# PART 3: SKILL EXTRACTION WITH ONTOLOGY
# ============================================================

print("PART 3: SKILL EXTRACTION VALIDATION")
print("-" * 70)

extractor = SkillExtractor()

extraction_test_cases = [
    {
        "name": "Extract ML skills",
        "text": "Expert in machine learning, scikit-learn, xgboost, and regression models",
        "expected_contains": ["machine learning", "scikit-learn", "xgboost"],
    },
    {
        "name": "Extract deep learning frameworks",
        "text": "Proficient in TensorFlow, PyTorch, and Keras",
        "expected_contains": ["tensorflow", "pytorch", "keras"],
    },
    {
        "name": "Extract data tools",
        "text": "Strong in pandas and numpy for data analysis",
        "expected_contains": ["pandas", "numpy"],
    },
]

extraction_passed = 0
for test in extraction_test_cases:
    skills = extractor.extract_skills(test["text"])
    success = all(skill in skills for skill in test["expected_contains"])
    
    status = "✅" if success else "❌"
    print(f"{status} {test['name']}")
    print(f"   Text: {test['text'][:60]}...")
    print(f"   Expected: {test['expected_contains']}")
    print(f"   Found: {[s for s in skills if s in test['expected_contains']]}")
    
    if success:
        extraction_passed += 1

print(f"\nExtraction Tests: {extraction_passed}/{len(extraction_test_cases)} PASSED\n")

# ============================================================
# PART 4: REALISTIC SCORING SCENARIOS
# ============================================================

print("PART 4: REALISTIC CANDIDATE SCENARIOS")
print("-" * 70)

# Mock components for scoring
class MockJobParser:
    def __init__(self):
        pass

# Create instances
skill_extractor = SkillExtractor()
semantic_matcher = SemanticMatcher()
role_detector = RoleDetector()
ontology_matcher = OntologyAwareSkillMatcher()

scenario_tests = [
    {
        "name": "Data Science candidate (strong match)",
        "resume_text": "Machine learning specialist with experience in scikit-learn, xgboost, pandas, python, sql, deep learning, tensorflow, nlp",
        "job_role": "Senior Data Scientist",
        "requirements": "machine learning, scikit-learn, python, sql, deep learning",
        "expected_min": 70,
        "expected_max": 100,
    },
    {
        "name": "ML Engineer (partial framework match)",
        "resume_text": "Experienced with deep learning, pytorch, tensorflow, neural networks, keras, python, nlp chatbot development",
        "job_role": "ML Engineer",
        "requirements": "deep learning, tensorflow, pytorch, python, nlp",
        "expected_min": 65,
        "expected_max": 100,
    },
    {
        "name": "Related skills candidate (ontology value)",
        "resume_text": "Data analysis, pandas, numpy, feature engineering, machine learning basics",
        "job_role": "Data Analyst",
        "requirements": "data analysis, pandas, numpy, machine learning",
        "expected_min": 60,  # Should be reasonable with partial matching
        "expected_max": 100,
    },
]

scenario_passed = 0
for scenario in scenario_tests:
    # Extract skills from resume and job
    resume_skills = skill_extractor.extract_skills(scenario["resume_text"])
    required_skills = skill_extractor.extract_from_requirements(scenario["requirements"])
    
    # Perform ontology matching
    match_result = ontology_matcher.match_skills(resume_skills, required_skills)
    skill_score = match_result["skill_match_score"]
    
    # Score should be in expected range
    score_normalized = skill_score * 100
    success = (scenario["expected_min"] <= score_normalized <= scenario["expected_max"]) or (score_normalized >= scenario["expected_min"])
    
    status = "✅" if success else "❌"
    print(f"{status} {scenario['name']}")
    print(f"   Resume: {len(resume_skills)} skills found")
    print(f"   Required: {required_skills}")
    print(f"   Matched: {len(match_result['exact_matches'])} exact, {len(match_result['partial_matches'])} partial")
    print(f"   Skill Score: {score_normalized:.1f}/100")
    
    if success:
        scenario_passed += 1

print(f"\nScenario Tests: {scenario_passed}/{len(scenario_tests)} PASSED\n")

# ============================================================
# FINAL SUMMARY
# ============================================================

total_passed = ontology_passed + matcher_passed + extraction_passed + scenario_passed
total_tests = len(test_cases_ontology) + len(matcher_test_cases) + len(extraction_test_cases) + len(scenario_tests)

print("="*70)
print(f"  FINAL RESULTS: {total_passed}/{total_tests} TESTS PASSED")
print("="*70)

if total_passed == total_tests:
    print("✅ ALL TESTS PASSED - Ontology system ready for integration!")
else:
    print(f"⚠️  {total_tests - total_passed} tests failed - review above")

print("="*70 + "\n")
