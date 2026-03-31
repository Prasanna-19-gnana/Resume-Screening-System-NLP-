"""
COMPLETE ML PIPELINE: Train and test learning-based ranking model

This script:
1. Imports all required components
2. Prepares training dataset
3. Trains RandomForest and XGBoost models
4. Evaluates on test set
5. Tests with 3-candidate ranking scenario

Usage:
    python train_ml_models.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required components
try:
    from app.services.skill_extractor_fixed import SkillExtractor
    from app.services.semantic_matcher_fixed import SemanticMatcher
    from app.services.role_detector_fixed import RoleDetector
    from app.services.context_matcher import ContextAwareMatcher
    from app.services.scorer_upgraded import UpgradedAIScorer
    from app.services.feature_engineering import FeatureEngineer
    from app.services.dataset_preparation import DatasetPreparer
    from app.services.model_training import MLModelTrainer
    from app.services.ml_scorer import MLScorer, EnsembleScorer
    from app.services.ranking_evaluator import RankingEvaluator
    
    logger.info("✅ All components imported successfully")
except Exception as e:
    logger.error(f"Failed to import components: {e}")
    sys.exit(1)


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("ML-BASED RESUME RANKING SYSTEM")
    print("="*60)
    
    # ===== PART 1: Initialize Components =====
    print("\n[1/6] Initializing components...")
    
    try:
        skill_extractor = SkillExtractor()
        semantic_matcher = SemanticMatcher()
        role_detector = RoleDetector()
        context_matcher = ContextAwareMatcher()
        
        logger.info("✅ All components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False
    
    # ===== PART 2: Feature Engineering =====
    print("\n[2/6] Setting up feature engineering...")
    
    try:
        feature_engineer = FeatureEngineer(
            skill_extractor, semantic_matcher, role_detector, context_matcher
        )
        
        logger.info(f"Features: {', '.join(feature_engineer.get_feature_names())}")
        logger.info("✅ Feature engineer ready")
    except Exception as e:
        logger.error(f"Feature engineering setup failed: {e}")
        return False
    
    # ===== PART 3: Dataset Preparation =====
    print("\n[3/6] Preparing training dataset...")
    
    try:
        dataset_preparer = DatasetPreparer(feature_engineer, None)
        
        # Create synthetic training dataset
        X, y = dataset_preparer.create_synthetic_dataset(n_samples=50)
        
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Score range: {y.min()*100:.1f} - {y.max()*100:.1f}")
        
        # Split into train/val/test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            dataset_preparer.split_dataset(X, y, test_size=0.2, validation_size=0.2)
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        logger.info("✅ Dataset prepared")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return False
    
    # ===== PART 4: Model Training =====
    print("\n[4/6] Training ML models...")
    
    try:
        trainer = MLModelTrainer(feature_engineer, None)
        
        # Train RandomForest
        print("\n  Training RandomForest...")
        rf_results = trainer.train_random_forest(X_train, y_train, X_val, y_val)
        print(f"  ✅ RandomForest trained: Val MAE={rf_results.get('val_mae', 0):.4f}")
        
        # Save RandomForest
        rf_path = trainer.save_model('rf')
        logger.info(f"RandomForest saved to {rf_path}")
        
        # Try XGBoost
        try:
            print("\n  Training XGBoost...")
            xgb_results = trainer.train_xgboost(X_train, y_train, X_val, y_val)
            
            if 'error' not in xgb_results:
                print(f"  ✅ XGBoost trained: Val MAE={xgb_results.get('val_mae', 0):.4f}")
                xgb_path = trainer.save_model('xgb')
                logger.info(f"XGBoost saved to {xgb_path}")
            else:
                print(f"  ⚠️  XGBoost not available: {xgb_results['error']}")
        except Exception as e:
            logger.warning(f"XGBoost training skipped: {e}")
        
        # Save results
        trainer.save_training_results(rf_results)
        logger.info("✅ Models trained and saved")
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False
    
    # ===== PART 5: ML Inference & Ranking =====
    print("\n[5/6] Testing ML scoring and ranking...")
    
    try:
        # Load trained model
        ml_scorer = MLScorer(feature_engineer, model_path=rf_path, fallback_scorer=None)
        
        # Test with 3 candidates
        test_candidates = [
            {
                "name": "NLP Engineer",
                "resume": "Python expert specializing in NLP, transformers, BERT, semantic search, " +
                          "deep learning with PyTorch, text embeddings, language models. " +
                          "5+ years experience in NLP projects and research."
            },
            {
                "name": "Full Stack Developer",
                "resume": "React and Node.js expert, REST APIs, MongoDB, some Python knowledge, " +
                         "basic ML exposure but no NLP specialization. " +
                         "3+ years full stack development."
            },
            {
                "name": "Junior Developer (No ML)",
                "resume": "C++ and Java developer, basic JavaScript, no Python, no machine learning, " +
                         "no NLP experience. 1 year professional experience."
            }
        ]
        
        job_description = """
        We are looking for an experienced NLP Engineer.
        Required skills: Python, NLP, transformers, BERT, semantic search, deep learning, PyTorch.
        """
        requirements = "Python, NLP, transformers, BERT, semantic search, deep learning, PyTorch"
        
        # Score all candidates
        print("\n  Scoring candidates with ML model...")
        ml_scores = []
        for candidate in test_candidates:
            result = ml_scorer.score_resume(
                resume_text=candidate["resume"],
                job_description=job_description,
                requirements=requirements,
                return_features=False
            )
            ml_scores.append(result["score"])
            print(f"    {candidate['name']}: {result['score']:.1f}")
        
        # Compute rankings
        ml_ranking = sorted(
            range(len(ml_scores)),
            key=lambda i: ml_scores[i],
            reverse=True
        )
        
        print("\n  ML Model Rankings:")
        for rank, idx in enumerate(ml_ranking, 1):
            print(f"    {rank}. {test_candidates[idx]['name']} ({ml_scores[idx]:.1f})")
        
        # Check if correct ranking
        expected_ranking = [0, 1, 2]  # NLP, Full Stack, Junior
        is_correct = ml_ranking == expected_ranking
        
        if is_correct:
            print("\n  ✅ CORRECT RANKING: NLP Engineer ranked #1 (as expected)")
        else:
            print("\n  ⚠️  Ranking differs from expected (may need more training data)")
        
        logger.info("✅ ML inference test completed")
    
    except Exception as e:
        logger.error(f"ML inference test failed: {e}")
        return False
    
    # ===== PART 6: Evaluation Metrics =====
    print("\n[6/6] Computing evaluation metrics...")
    
    try:
        evaluator = RankingEvaluator()
        
        # Evaluate ML model on test set
        y_pred_test = trainer.rf_model.predict(X_test)
        y_pred_test = np.clip(y_pred_test * 100, 0, 100)  # Scale to 0-100
        
        metrics = evaluator.evaluate_ranking(
            predicted_scores=list(y_pred_test),
            ground_truth_scores=list(y_test * 100),
            k_values=[1, 5, 10]
        )
        
        evaluator.print_metrics(metrics)
        logger.info("✅ Evaluation completed")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False
    
    # ===== Summary =====
    print("\n" + "="*60)
    print("ML PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"""
Next steps:
1. Integrate MLScorer into app.py
2. Use ml_score in multi_resume_ranker.py instead of rule-based scoring
3. Monitor model performance in production

Model File: {rf_path}
Feature Count: 7
Training Samples: {X_train.shape[0]}
Model Type: RandomForest
""")
    
    return True


if __name__ == "__main__":
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)
