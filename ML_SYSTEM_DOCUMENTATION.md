# ML-Based Resume Ranking System

## Overview

This document describes the complete machine learning-based resume ranking system that replaces manual rule-based scoring with trained models.

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│                  (Resume + Job Description)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering                         │
│  (Extract 7 numerical features from resume-JD pair)        │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      ┌───────┐    ┌──────────┐   ┌─────────┐
      │ ML    │ +  │  Rule-   │ + │Semantic │
      │Model  │    │  Based   │   │  Match  │
      └───────┘    └──────────┘   └─────────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │   Final Score (0-100)     │
          │   Recommendation          │
          │   Confidence score        │
          └──────────────────────────┘
```

## Feature Set (7 Features)

| # | Feature | Type | Range | Description |
|---|---------|------|-------|-------------|
| 1 | `semantic_similarity_score` | float | 0.0-1.0 | Semantic match b/w resume & JD using embeddings |
| 2 | `skills_exact_match_ratio` | float | 0.0-1.0 | Proportion of required skills with exact matches |
| 3 | `skills_partial_match_ratio` | float | 0.0-1.0 | Proportion of required skills with partial matches |
| 4 | `role_alignment_score` | float | 0.0-1.0 | How well detected role matches job role |
| 5 | `number_of_matched_skills` | int | 0-N | Count of exactly matched skills |
| 6 | `number_of_missing_skills` | int | 0-N | Count of required but missing skills |
| 7 | `top_sentence_similarity` | float | 0.0-1.0 | Highest semantic similarity of any sentence pair |

## Models

### RandomForest (Primary)
- **Type**: RandomForestRegressor
- **Estimators**: 100
- **Max Depth**: 10
- **Location**: `ai_resume_screening/models/ml_scorer_rf.pkl`
- **Feature Importances**:
  - `number_of_missing_skills`: 57.0%
  - `skills_exact_match_ratio`: 28.0%
  - `number_of_matched_skills`: 15.0%
  - Others: 0%

### XGBoost (Optional)
- **Type**: XGBRegressor
- **Status**: Optional (install with `pip install xgboost`)
- **Location**: `ai_resume_screening/models/ml_scorer_xgb.pkl` (if trained)

## Scoring Framework

### Three Modes

#### 1. **ML-Only** (Pure ML)
```
final_score = ML_model.predict(features)
```
Best for: Maximum accuracy (requires good training data)

#### 2. **Hybrid** (Recommended - Default)
```
final_score = 0.70 * ml_score + 0.30 * rule_based_score
```
Best for: Balance between ML and explainability

#### 3. **Fallback** (Rule-Based Only)
```
final_score = rule_based_scorer(resume, jd)
```
Best for: Safety/reliability when ML model unavailable

### Formula Comparison

**Rule-Based (Old)**:
```
score = (Semantic × 0.35) + (Skills × 0.45) + (Role × 0.20)
```

**ML-Based (New)**:
```
score = RandomForest.predict(
  [semantic_sim, exact_match, partial_match, role_align, 
   matched_count, missing_count, top_sent_sim]
)
```

The ML model automatically learns optimal weighting from data.

## API Usage

### Python Code

```python
from app.services.ml_scorer_service import ml_scorer_service

# Score a resume
result = ml_scorer_service.score_candidate(
    parsed_resume_dict={
        "summary": "...",
        "skills": "Python, ML, NLP",
        "experience": "...",
        # ... other sections
    },
    job_role="NLP Engineer",
    job_description="Looking for NLP expert...",
    requirements="Python, NLP, BERT, transformers",
    resume_semantic_text="..."
)

# Result contains:
final_score = result["final_score"]  # 0.0-1.0
recommendation = result["recommendation"]  # "Strong Match", etc
method = result["scoring_method"]  # "ml", "hybrid", or "fallback"

if "ml_score" in result:
    ml_score = result["ml_score"]
    rule_score = result["rule_score"]
    print(f"ML: {ml_score:.3f}, Rule: {rule_score:.3f}")
```

### REST API

Update app/main.py to support scoring mode:

```python
@app.post("/match")
async def screen_candidate(req: MatchRequest, mode: str = "hybrid"):
    # mode can be "ml", "hybrid", or "fallback"
    service = select_scorer(mode)
    return service.score_candidate(...)
```

## Training Pipeline

### 1. Feature Engineering
```
Resume + JD → Extract 7 features → Feature vector [7]
```

### 2. Dataset Preparation
```
50 synthetic samples from predefined test cases
X: (50, 7) feature matrix
y: (50,) score vector (0-1 normalized)
```

### 3. Model Training
```
Train/Val/Test split: 60% / 20% / 20%
Training results saved to training_results.json
Model saved to ml_scorer_rf.pkl
```

### 4. Evaluation
```
RandomForest Metrics:
- Train MAE: 0.15
- Val MAE: 0.19
- Train R²: 0.43
- Val R²: 0.21
```

### 5. Production Deployment
```
Model automatically loaded when service initializes
Falls back to rule-based if model not found
```

## File Structure

```
ai_resume_screening/
├── app/
│   ├── main.py                          # FastAPI app
│   └── services/
│       ├── feature_engineering.py       # Feature extraction
│       ├── ml_scorer.py                 # ML inference
│       ├── ml_scorer_service.py         # Production service (NEW)
│       ├── dataset_preparation.py       # Training data prep
│       ├── model_training.py            # Model training
│       ├── ranking_evaluator.py         # Metrics
│       └── scorer.py                    # Rule-based (legacy)
│
├── models/
│   ├── ml_scorer_rf.pkl                 # Trained RandomForest
│   ├── ml_scorer_xgb.pkl                # Trained XGBoost (optional)
│   └── training_results.json            # Training metrics
│
├── training/
│   └── train_ml_models.py               # Training script
│
└── test_ml_integration.py               # Integration test
```

## Performance

### Test Case: 3 Candidates

| Candidate | Score | Rank | Method |
|-----------|-------|------|--------|
| NLP Engineer | 91.6 | 1 ✅ | ML |
| Full Stack Dev | 74.2 | 2 | ML |
| Junior Dev | 74.2 | 3 | ML |

**Ranking**: ✅ CORRECT (NLP engineer ranked #1)

### Evaluation Metrics

```
Precision@1: 1.0000 (100%)
Recall@1: 0.2000 (20%)
NDCG@1: 1.0000 (100%)
NDCG@5: 0.8539 (85.4%)
NDCG@10: 0.9520 (95.2%)
MRR: 1.0000
AP: 0.8600
```

## How to Use

### 1. Training (One-time Setup)

```bash
cd ai_resume_screening
python training/train_ml_models.py
```

This:
- Creates synthetic training data
- Trains RandomForest model
- Saves model to `models/ml_scorer_rf.pkl`
- Generates training_results.json

### 2. Initialization (Automatic)

```python
from app.services.ml_scorer_service import ml_scorer_service

# Model automatically loaded on first use
# Falls back gracefully if model not found
```

### 3. Scoring

```python
result = ml_scorer_service.score_candidate(...)

# Three possible scoring methods returned:
# - "ml": Pure ML (when mode="ml")
# - "hybrid": ML + rule-based blend (when mode="hybrid")  
# - "fallback": Rule-based only (when model unavailable)
```

### 4. Monitoring

```python
model_info = ml_scorer_service.get_model_info()
print(model_info)
# Output:
# {
#   "mode": "hybrid",
#   "ml_weight": 0.7,
#   "ml_model_available": True,
#   "ml_model_info": {...}
# }
```

## Configuration

### Change Scoring Mode

```python
from app.services.ml_scorer_service import MLScorerService

# Pure ML
ml_service = MLScorerService(mode="ml")

# Hybrid (70% ML, 30% rule-based)
hybrid_service = MLScorerService(mode="hybrid", ml_weight=0.7)

# Fallback (rule-based only)
fallback_service = MLScorerService(mode="fallback")
```

### Change ML Weight in Hybrid Mode

```python
# 80% ML, 20% rule-based
service = MLScorerService(mode="hybrid", ml_weight=0.8)
```

## Extending the System

### Adding More Features

1. Update `FeatureEngineer.extract_features()` to compute new feature
2. Update `FeatureEngineer.get_feature_names()` with new name
3. Retrain: `python training/train_ml_models.py`

### Using Real Labeled Data

Instead of synthetic data:

```python
# In dataset_preparation.py
samples = load_your_labeled_data()  # List of resume-JD-score tuples
X, y = dataset_preparer.prepare_dataset_from_samples(samples)
```

### Switching to Different Model

```python
# In model_training.py, replace RandomForest with:
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(...)
model.fit(X_train, y_train)
```

## Troubleshooting

### Model Not Loading

```
Error: ML model not loaded, using fallback

Solution: Run training script first
$ python training/train_ml_models.py
```

### Low Accuracy

Possible causes:
- Insufficient training data (use real labeled data)
- Feature scaling issues (normalize features)
- Model overfitting (reduce depth, increase samples)

### Features Not Being Computed

Check feature_engineering.py fallback values:
```python
def _get_default_features(self):
    return {
        "semantic_similarity_score": 0.5,
        # ... all defaulting to 0.5 or 0.0
    }
```

## Next Steps

1. **Collect Real Data**: Gather labeled resume-JD-score pairs
2. **Retrain**: Use real data for model training
3. **A/B Testing**: Compare ML vs rule-based scoring
4. **Monitor**: Track model performance over time
5. **Iterate**: Improve features, models, hyperparameters

## References

- [Feature Engineering Module](../app/services/feature_engineering.py)
- [ML Scorer Module](../app/services/ml_scorer.py)
- [ML Scorer Service](../app/services/ml_scorer_service.py)
- [Training Pipeline](../training/train_ml_models.py)
- [Ranking Evaluator](../app/services/ranking_evaluator.py)
