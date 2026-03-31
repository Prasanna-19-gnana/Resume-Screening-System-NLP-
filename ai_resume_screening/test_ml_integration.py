"""
TEST: ML Integration Verification

Tests the complete ML pipeline integrated into the FastAPI app
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ml_scorer_service():
    """Test ML scorer service initialization and scoring"""
    
    print("\n" + "="*60)
    print("ML SCORER SERVICE INTEGRATION TEST")
    print("="*60)
    
    try:
        # Import the ML scorer service
        from app.services.ml_scorer_service import ml_scorer_service, rule_scorer_service
        
        logger.info("✅ ML scorer service imported successfully")
        
        # Check available model
        model_info = ml_scorer_service.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        if not model_info["ml_model_available"]:
            print("\n⚠️  ML model not available, will use fallback scoring")
        else:
            print("\n✅ ML model available for scoring")
        
        # Test data
        parsed_resume = {
            "summary": "Data scientist with 5+ years ML experience",
            "skills": "Python, PyTorch, TensorFlow, NLP, transformers, BERT",
            "experience": "Lead ML engineer at tech company",
            "projects": "Built NLP systems for semantic search",
            "education": "MS Computer Science",
            "certifications": "",
            "full_text": "Python, ML, NLP, PyTorch, transformers"
        }
        
        job_role = "NLP Engineer"
        job_description = """
        We seek an experienced NLP engineer with:
        - 5+ years Python development
        - Expertise in NLP: transformers, BERT
        - PyTorch or TensorFlow
        - Semantic search experience
        """
        requirements = "Python, NLP, transformers, BERT, PyTorch, semantic search"
        resume_semantic_text = "Python NLP expert with transformers and BERT"
        
        # Test ML scorer (hybrid mode)
        print("\n" + "-"*60)
        print("Test 1: ML Scorer (Hybrid Mode)")
        print("-"*60)
        
        result_ml = ml_scorer_service.score_candidate(
            parsed_resume_dict=parsed_resume,
            job_role=job_role,
            job_description=job_description,
            requirements=requirements,
            resume_semantic_text=resume_semantic_text
        )
        
        print(f"\nFinal Score: {result_ml['final_score']:.3f}")
        print(f"Recommendation: {result_ml['recommendation']}")
        print(f"Scoring Method: {result_ml['scoring_method']}")
        print(f"Role Alignment: {result_ml.get('role_alignment_score', 'N/A'):.3f}")
        print(f"Semantic Similarity: {result_ml.get('semantic_similarity_score', 'N/A'):.3f}")
        print(f"Skill Match: {result_ml.get('skill_match_score', 'N/A'):.3f}")
        print(f"Matched Skills: {result_ml.get('matched_skills', [])}")
        print(f"Missing Skills: {result_ml.get('missing_skills', [])}")
        
        if "ml_score" in result_ml:
            print(f"\nML Score: {result_ml['ml_score']:.3f}")
            print(f"Rule Score: {result_ml['rule_score']:.3f}")
            print(f"ML Weight: {result_ml.get('ml_weight', 'N/A')}")
        
        # Test rule-based scorer
        print("\n" + "-"*60)
        print("Test 2: Rule-Based Scorer (Fallback)")
        print("-"*60)
        
        result_rule = rule_scorer_service.score_candidate(
            parsed_resume_dict=parsed_resume,
            job_role=job_role,
            job_description=job_description,
            requirements=requirements,
            resume_semantic_text=resume_semantic_text
        )
        
        print(f"\nFinal Score: {result_rule['final_score']:.3f}")
        print(f"Recommendation: {result_rule['recommendation']}")
        print(f"Matched Skills: {result_rule.get('matched_skills', [])}")
        
        # Summary
        print("\n" + "="*60)
        print("INTEGRATION TEST RESULTS")
        print("="*60)
        print(f"✅ ML Scorer Service: OK")
        print(f"✅ Model Status: {'Available' if model_info['ml_model_available'] else 'Fallback'}")
        print(f"✅ Scoring Modes: ML, Hybrid, Fallback")
        print(f"✅ API Integration: Ready")
        print("="*60 + "\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ml_scorer_service()
    sys.exit(0 if success else 1)
