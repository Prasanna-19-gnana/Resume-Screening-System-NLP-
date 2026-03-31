"""
TEST: Multi-Resume Ranking and Context-Aware Matching
Test with 3 sample resumes for NLP Engineer position
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
from context_matcher import ContextAwareMatcher
from multi_resume_ranker import MultiResumeRanker

print("\n" + "="*80)
print("  MULTI-RESUME RANKING TEST")
print("  Context-Aware Matching with 3 Sample Candidates")
print("="*80 + "\n")

# ============================================================
# PART 1: DEFINE TEST RESUMES
# ============================================================

nlp_engineer_resume = """
NLP Engineer - Senior

SUMMARY:
Experienced NLP specialist with 6 years of expertise in natural language processing, 
transformer models, and text analysis. Proven track record building production NLP systems.

SKILLS:
Natural Language Processing, Transformers, BERT, GPT, Python, PyTorch, TensorFlow, 
NLTK, SpaCy, Text Classification, Named Entity Recognition, Sentiment Analysis, 
Word Embeddings, Language Models, Deep Learning, CUDA, Linux

EXPERIENCE:
Senior NLP Engineer at TechCorp (2022-Present)
- Developed transformer-based NLP models for text classification
- Optimized BERT fine-tuning pipeline achieving 95% accuracy
- Implemented named entity recognition system for document extraction
- Managed team of 2 junior NLP engineers

NLP Research Scientist at AILabs (2020-2022)
- Published 3 papers on transformer models in peer-reviewed venues
- Built GPT-2 fine-tuning framework for domain adaptation
- Conducted sentiment analysis research on large-scale datasets

Python Developer at DataCorp (2018-2020)
- Worked with NLTK and SpaCy for text processing
- Built NLP pipelines for customer support automation
- Optimized text processing performance using CUDA

EDUCATION:
MS Computer Science, specializing in NLP
BS Mathematics with Computer Science minor

PROJECTS:
- Developed multilingual sentiment analyzer using transformers
- Created automated document classification system using BERT
- Built word embedding visualization tool using TensorFlow
"""

fullstack_engineer_resume = """
Full Stack Engineer

SUMMARY:
Experienced full stack developer with 5 years building web applications. 
Proficient in frontend and backend technologies.

SKILLS:
React, Node.js, JavaScript, Python, MongoDB, PostgreSQL, Docker, AWS, 
HTML5, CSS3, REST APIs, Git, Linux, Django, Express.js, Machine Learning basics

EXPERIENCE:
Senior Full Stack Engineer at WebCo (2021-Present)
- Built React applications with Node.js backends
- Designed and optimized PostgreSQL databases
- Deployed applications to AWS using Docker

Full Stack Developer at StartupXYZ (2019-2021)
- Developed web applications using MERN stack
- Implemented REST APIs with Express.js
- Worked with MongoDB for NoSQL solutions

Junior Developer at TechStartup (2018-2019)
- Built responsive websites with React and CSS
- Used Python for backend scripting
- Learned cloud deployment with AWS

EDUCATION:
BS Computer Science

PROJECTS:
- Built social media platform with React and Node.js
- Created e-commerce system with Django and React
- Deployed microservices using Docker and Kubernetes
"""

non_ml_candidate_resume = """
Business Analyst

SUMMARY:
Business analyst with 4 years of experience in requirements gathering and data analysis.

SKILLS:
Excel, SQL, Business Intelligence, Data Analysis, PowerPoint, Microsoft Office, 
Project Management, Tableau, Python basics, Statistics

EXPERIENCE:
Senior Business Analyst at CompanyABC (2021-Present)
- Gathered and documented business requirements
- Created dashboards using Tableau
- Analyzed sales data using Excel and SQL
- Generated reports for stakeholder presentations

Business Analyst at RetailCorp (2019-2021)
- Collected requirements from business stakeholders
- Performed data analysis in Excel and SQL
- Created business intelligence reports
- Managed project timelines and documentation

EDUCATION:
BA Business Administration
MBA in progress

PROJECTS:
- Built sales dashboard in Tableau
- Analyzed customer data for business insights
- Documented system requirements for IT team
"""

# ============================================================
# PART 2: SETUP JOB DESCRIPTION
# ============================================================

job_role = "Senior NLP Engineer"
job_description = """
We are seeking a Senior NLP Engineer to join our AI team.

RESPONSIBILITIES:
- Develop and train transformer-based natural language processing models
- Implement text classification and named entity recognition systems
- Optimize deep learning pipelines for production deployment
- Research and evaluate state-of-the-art NLP techniques
- Collaborate with data scientists and ML engineers

REQUIRED QUALIFICATIONS:
- 5+ years of NLP experience
- Strong Python and PyTorch/TensorFlow expertise
- Proven experience with transformer models (BERT, GPT, RoBERTa)
- Deep understanding of NLP tasks: classification, NER, sentiment analysis
- Experience with production machine learning systems
- Linux and CUDA optimization skills

NICE TO HAVE:
- Published NLP research
- Experience with multilingual models
- Knowledge of distributed training
- Cloud platform experience

COMPENSATION:
Competitive salary, benefits, remote-friendly
"""

requirements = "transformers, NLP, Python, PyTorch, TensorFlow, BERT, deep learning, text classification, named entity recognition"

# ============================================================
# PART 3: INITIALIZE COMPONENTS
# ============================================================

print("Initializing components...")

skill_extractor = SkillExtractor()
semantic_matcher = SemanticMatcher()
role_detector = RoleDetector()
ontology_matcher = OntologyAwareSkillMatcher()
context_matcher = ContextAwareMatcher()

scorer = create_upgraded_scorer(
    skill_extractor=skill_extractor,
    semantic_matcher=semantic_matcher,
    role_detector=role_detector,
    ontology_matcher=ontology_matcher
)

ranker = MultiResumeRanker(
    scorer=scorer,
    context_matcher=context_matcher,
    skill_extractor=skill_extractor
)

print("✅ Components loaded\n")

# ============================================================
# PART 4: PREPARE RESUME DATA
# ============================================================

resumes_data = [
    {"name": "nlp_engineer.pdf", "text": nlp_engineer_resume},
    {"name": "fullstack_engineer.pdf", "text": fullstack_engineer_resume},
    {"name": "business_analyst.pdf", "text": non_ml_candidate_resume},
]

# ============================================================
# PART 5: RANK RESUMES
# ============================================================

print("Ranking resumes...\n")

ranking_result = ranker.rank_resumes(
    resumes_data=resumes_data,
    job_role=job_role,
    requirements=requirements,
    job_description=job_description
)

# ============================================================
# PART 6: DISPLAY RESULTS
# ============================================================

print("="*80)
print("  RANKING RESULTS")
print("="*80 + "\n")

print(f"Position: {job_role}")
print(f"Total Resumes Evaluated: {ranking_result['total_resumes']}\n")

print("-"*80)
print(f"SUMMARY:")
print(f"  Top Candidate: {ranking_result['summary']['top_candidate']}")
print(f"  Top Score: {ranking_result['summary']['top_score']:.1f}/100")
print(f"  Average Score: {ranking_result['summary']['average_score']:.1f}/100")
print(f"  {ranking_result['summary']['recommendation']}\n")

print("-"*80)
print("RANKED LIST:\n")

for resume in ranking_result['ranked_results']:
    print(f"RANK #{resume['rank']}: {resume['resume_name']}")
    print(f"  Final Score: {resume['final_score']:.1f}/100")
    print(f"  Recommendation: {resume['recommendation']}")
    print(f"  Confidence: {resume['confidence']}")
    
    print(f"\n  SCORE BREAKDOWN:")
    print(f"    - Semantic Match: {resume['semantic_score']:.1f}%")
    print(f"    - Skill Match: {resume['skill_score']:.1f}%")
    print(f"    - Role Alignment: {resume['role_score']:.1f}%")
    
    print(f"\n  SKILLS ANALYSIS:")
    print(f"    - Matched: {resume['num_skills_matched']} / {resume['num_skills_required']} ({resume['skills_coverage']})")
    print(f"    - Matched Skills: {resume['matched_skills'][:3]}")  # Show first 3
    
    if resume.get('missing_skills'):
        print(f"    - Missing: {resume['missing_skills'][:3]}")
    
    print(f"\n  EVIDENCE (Context-Aware Matching):")
    if resume['strong_evidence']:
        for i, evidence in enumerate(resume['strong_evidence'][:2], 1):
            print(f"    [{i}] {evidence['sentence'][:80]}...")
            print(f"        Relevance: {evidence['relevance']}")
    else:
        print(f"    No strong matching evidence found")
    
    print(f"\n  Coverage: {resume['coverage_score']} of resume aligns with job description")
    print(f"  Summary: {resume['evidence_summary']}")
    
    print("-"*80 + "\n")

# ============================================================
# PART 7: VALIDATION
# ============================================================

print("="*80)
print("  VALIDATION")
print("="*80 + "\n")

# Check expected ranking
ranked_names = [r['resume_name'] for r in ranking_result['ranked_results']]
expected_order = ["nlp_engineer.pdf", "fullstack_engineer.pdf", "business_analyst.pdf"]

print("Expected Ranking Order:")
for i, name in enumerate(expected_order, 1):
    print(f"  {i}. {name}")

print("\nActual Ranking Order:")
for i, name in enumerate(ranked_names, 1):
    print(f"  {i}. {name}")

# Validate
validation_passed = True
if ranked_names[0] != "nlp_engineer.pdf":
    print("\n❌ VALIDATION FAILED: NLP Engineer should rank #1")
    validation_passed = False
else:
    print("\n✅ Rank 1: NLP Engineer (CORRECT)")

if ranked_names[1] != "fullstack_engineer.pdf":
    print("❌ VALIDATION FAILED: Full Stack should rank #2")
    validation_passed = False
else:
    print("✅ Rank 2: Full Stack Engineer (CORRECT)")

if ranked_names[2] != "business_analyst.pdf":
    print("❌ VALIDATION FAILED: Business Analyst should rank #3")
    validation_passed = False
else:
    print("✅ Rank 3: Business Analyst (CORRECT)")

print("\n" + "="*80)
if validation_passed:
    print("  ✅ ALL VALIDATION TESTS PASSED")
    print("  Multi-resume ranking working correctly!")
else:
    print("  ⚠️ Some validation tests failed")
print("="*80 + "\n")

# Export CSV
print("\nExporting ranking to CSV format...")
csv_output = ranker.export_ranking_csv(ranking_result)
print("\nCSV Export Preview:")
print(csv_output[:500])
print("...")
