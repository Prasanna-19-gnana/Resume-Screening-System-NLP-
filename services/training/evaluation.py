import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    ndcg_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

class EvaluationMetrics:
    """Helper methods for calculating Ranking metrics."""
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculates Precision@K. Relevant items are non-zero true labels."""
        order = np.argsort(y_pred)[::-1]
        top_k = y_true[order][:k]
        return np.sum(top_k > 0) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculates Recall@K."""
        order = np.argsort(y_pred)[::-1]
        top_k = y_true[order][:k]
        total_relevant = np.sum(y_true > 0)
        return np.sum(top_k > 0) / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def reciprocal_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates MRR for a single query."""
        order = np.argsort(y_pred)[::-1]
        top_y = y_true[order]
        relevant_indices = np.where(top_y > 0)[0]
        if len(relevant_indices) == 0:
            return 0.0
        return 1.0 / (relevant_indices[0] + 1)


class ModelEvaluator:
    """
    Comprehensive evaluation module handling Classification, Regression, 
    Ranking, Model Comparisons, and Error Analysis.
    """
    def __init__(self, data_path: str, model_dir: str):
        self.data_path = data_path
        self.model_dir = model_dir
        self.feature_cols = [
            "req_skill_coverage", "pref_skill_coverage", "total_matched_skills",
            "missing_req_skills", "semantic_similarity", "resume_word_count"
        ]
        
        # Label encodings for string -> int map to simplify metric calculations
        self.label_map = {"poor": 0, "medium": 1, "good": 2}
        self.reverse_label_map = {0: "poor", 1: "medium", 2: "good"}

    def load_data(self) -> pd.DataFrame:
        """Loads dataset and features."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rows = []
        for sample in data:
            if "features" not in sample: continue
            row = sample["features"].copy()
            row["fit_label"] = sample.get("fit_label", "poor")
            row["fit_score"] = sample.get("fit_score_true", 0)
            
            # Context id defines a 'query' for ranking. We use Job Description natively.
            row["job_id"] = hash(sample.get("job_description", ""))
            row["id"] = sample.get("id", "")
            rows.append(row)
            
        return pd.DataFrame(rows)

    def evaluate_classification(self, model, X: pd.DataFrame, y_true_labels: pd.Series) -> Dict[str, Any]:
        """Runs standard classification metrics."""
        y_pred = model.predict(X)
        y_true_encoded = y_true_labels.map(self.label_map)
        y_pred_encoded = pd.Series(y_pred).map(self.label_map)
        
        # Using macro to treat each class equally even if imbalanced
        metrics = {
            "accuracy": accuracy_score(y_true_encoded, y_pred_encoded),
            "precision_macro": precision_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true_encoded, y_pred_encoded).tolist(),
            "classification_report": classification_report(y_true_encoded, y_pred_encoded, zero_division=0, output_dict=True)
        }
        return metrics

    def evaluate_regression(self, model, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """Runs continuous variable metrics for fit score."""
        y_pred = model.predict(X)
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred)
        }
        return metrics

    def evaluate_ranking(self, model, df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
        """Runs rank-aware metrics (NDCG, MRR, P@K) grouped by Job Description."""
        # For ranking we use probability of the 'good' class or a regression score
        if hasattr(model, "predict_proba"):
            X = df[self.feature_cols]
            # Assuming class 'good' is the last index if strictly sorted alphabetically
            # To be safe, look at model.classes_
            classes = list(model.classes_)
            good_idx = classes.index("good") if "good" in classes else -1
            scores = model.predict_proba(X)[:, good_idx]
        else:
            scores = model.predict(df[self.feature_cols])

        df["pred_score"] = scores
        df["relevance"] = df["fit_label"].map(self.label_map)

        ndcg_list, mrr_list, p_at_k_list, r_at_k_list = [], [], [], []
        
        for job_id, group in df.groupby("job_id"):
            y_true = group["relevance"].values
            y_pred = group["pred_score"].values
            
            if len(y_true) < 2:
                continue # Ranking metrics need at least 2 items to form a list
                
            # NDCG from scikit-learn expects 2D array [n_samples, n_items]
            ndcg_val = ndcg_score([y_true], [y_pred], k=k)
            ndcg_list.append(ndcg_val)
            
            p_val = EvaluationMetrics.precision_at_k(y_true, y_pred, k)
            p_at_k_list.append(p_val)
            
            r_val = EvaluationMetrics.recall_at_k(y_true, y_pred, k)
            r_at_k_list.append(r_val)
            
            mrr_val = EvaluationMetrics.reciprocal_rank(y_true, y_pred)
            mrr_list.append(mrr_val)
            
        if not ndcg_list:
            return {"mean_ndcg": 0.0, "mrr": 0.0, f"precision@{k}": 0.0, f"recall@{k}": 0.0}
            
        return {
            "mean_ndcg": np.mean(ndcg_list),
            "mrr": np.mean(mrr_list),
            f"precision@{k}": np.mean(p_at_k_list),
            f"recall@{k}": np.mean(r_at_k_list)
        }

    def error_analysis(self, model, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Isolates False Positives, False Negatives, and Mis-ranked entries."""
        X = df[self.feature_cols]
        y_true = df["fit_label"]
        y_pred = model.predict(X)
        
        false_positives = []
        false_negatives = []
        
        for i in range(len(df)):
            true_l = y_true.iloc[i]
            pred_l = y_pred[i]
            sample_id = df.iloc[i]["id"]
            row_dict = {"id": sample_id, "true": true_l, "pred": pred_l, "features": df.iloc[i][self.feature_cols].to_dict()}
            
            # False Positive: Model predicts 'good', True is 'poor'
            if pred_l == "good" and true_l == "poor":
                false_positives.append(row_dict)
            
            # False Negative: Model predicts 'poor', True is 'good'
            elif pred_l == "poor" and true_l == "good":
                false_negatives.append(row_dict)
                
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    def baseline_scoring(self, df: pd.DataFrame) -> np.ndarray:
        """Rule-based naive scoring system to use as baseline comparison."""
        scores = []
        for i, row in df.iterrows():
            # Old manual rule: 60% semantic, 20% required coverage, 20% total skills proxy
            # Returns a string label
            score = (0.6 * row["semantic_similarity"]) + (0.3 * row["req_skill_coverage"]) + (0.1 * (min(row["total_matched_skills"]/10.0, 1.0)))
            if score > 0.7: scores.append("good")
            elif score > 0.4: scores.append("medium")
            else: scores.append("poor")
        return np.array(scores)

    def generate_full_report(self):
        """Orchestrates comparisons, calculations and handles IO."""
        df = self.load_data()
        X = df[self.feature_cols]
        y_class = df["fit_label"]
        y_reg = df["fit_score"]

        # Load existing RF models
        rf_class_path = os.path.join(self.model_dir, "rf_classifier.pkl")
        rf_reg_path = os.path.join(self.model_dir, "rf_regressor.pkl")
        
        rf_clf = joblib.load(rf_class_path) if os.path.exists(rf_class_path) else None
        rf_reg = joblib.load(rf_reg_path) if os.path.exists(rf_reg_path) else None

        # Train a Logistic Regression instance purely for evaluation comparison
        lr_clf = LogisticRegression(max_iter=500, class_weight='balanced')
        if len(df) > 1: lr_clf.fit(X, y_class)

        report = {
            "dataset_size": len(df),
            "models": {}
        }

        # 1. Evaluate RandomForest
        if rf_clf:
            report["models"]["RandomForest"] = {
                "classification": self.evaluate_classification(rf_clf, X, y_class),
                "ranking": self.evaluate_ranking(rf_clf, df, k=5),
                "error_analysis": self.error_analysis(rf_clf, df)
            }
        
        if rf_reg:
            report["models"]["RandomForest"]["regression"] = self.evaluate_regression(rf_reg, X, y_reg)

        # 2. Evaluate Logistic Regression Comparitor
        if len(df) > 1:
            report["models"]["LogisticRegression"] = {
                "classification": self.evaluate_classification(lr_clf, X, y_class),
                "ranking": self.evaluate_ranking(lr_clf, df, k=5),
            }

        # 3. Baseline Rule-Based Model
        baseline_preds = self.baseline_scoring(df)
        b_true_enc = y_class.map(self.label_map)
        b_pred_enc = pd.Series(baseline_preds).map(self.label_map)
        report["models"]["BaselineRuleBased"] = {
            "classification": {
                "accuracy": accuracy_score(b_true_enc, b_pred_enc),
                "precision_macro": precision_score(b_true_enc, b_pred_enc, average="macro", zero_division=0),
                "recall_macro": recall_score(b_true_enc, b_pred_enc, average="macro", zero_division=0),
                "f1_macro": f1_score(b_true_enc, b_pred_enc, average="macro", zero_division=0)
            }
        }

        # 4. Generate the Output JSON
        output_file = os.path.join(self.model_dir, "evaluation_report.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        self.print_summary(report)

    def print_summary(self, report: Dict[str, Any]):
        """Prints formatted metrics to terminal."""
        print("="*60)
        print("   RESUME SCREENING SYSTEM - EVALUATION REPORT")
        print("="*60)
        print(f"Dataset Items Evaluated: {report['dataset_size']}\n")
        
        for model_name, metrics in report["models"].items():
            print(f"--- MODEL: {model_name} ---")
            
            if "classification" in metrics:
                clf_m = metrics["classification"]
                if "classification_report" in clf_m:
                    print(f"Accuracy: \t{clf_m['accuracy']:.4f}")
                    print(f"F1 (Macro): \t{clf_m['f1_macro']:.4f}")
                else: 
                     # Baseline print
                    print(f"Accuracy: \t{clf_m['accuracy']:.4f}")
                    print(f"F1 (Macro): \t{clf_m['f1_macro']:.4f}")

            if "regression" in metrics:
                reg_m = metrics["regression"]
                print(f"RMSE: \t\t{reg_m['rmse']:.4f}")
                print(f"MAE: \t\t{reg_m['mae']:.4f}")
                print(f"R2: \t\t{reg_m['r2_score']:.4f}")

            if "ranking" in metrics:
                rk_m = metrics["ranking"]
                print(f"NDCG@5: \t{rk_m.get('mean_ndcg', 0.0):.4f}")
                print(f"MRR: \t\t{rk_m.get('mrr', 0.0):.4f}")
                print(f"Precision@5: \t{rk_m.get('precision@5', 0.0):.4f}")
            
            print()

        # Print Error Analysis Summary for the Best Model (RF)
        if "RandomForest" in report["models"] and "error_analysis" in report["models"]["RandomForest"]:
            errs = report["models"]["RandomForest"]["error_analysis"]
            fp, fn = len(errs["false_positives"]), len(errs["false_negatives"])
            print(f"Error Analysis (RandomForest):")
            print(f"- False Positives ('Poor' rated 'Good'): {fp}")
            print(f"- False Negatives ('Good' rated 'Poor'): {fn}")
        print("="*60)
        print(f"Full metrics saved to: {os.path.join(self.model_dir, 'evaluation_report.json')}")

# ---------------------------------------------------------
# Usage example: Run Full Evaluation Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(base_dir, "data/training_dataset_with_features.json")
    model_dir = os.path.join(base_dir, "models/")
    
    evaluator = ModelEvaluator(data_path, model_dir)
    print("Starting Comprehensive Evaluation Process...")
    evaluator.generate_full_report()
