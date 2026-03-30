import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error

class ModelTrainer:
    """
    Trains and saves machine learning models for predicting resume fit.
    """
    def __init__(self, data_path: str, model_dir: str):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Feature columns defined in feature_engineering.py
        self.feature_cols = [
            "req_skill_coverage",
            "pref_skill_coverage",
            "total_matched_skills",
            "missing_req_skills",
            "semantic_similarity",
            "resume_word_count"
        ]

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Loads features and targets from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rows = []
        for sample in data:
            if "features" not in sample:
                continue
            row = sample["features"].copy()
            # Targets
            row["fit_label"] = sample.get("fit_label", "unknown")
            row["fit_score"] = sample.get("fit_score_true", 0)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No valid features found in dataset.")
            
        # Filter out unknown labels if any
        df = df[df["fit_label"] != "unknown"]
        
        X = df[self.feature_cols]
        y_class = df["fit_label"]
        y_reg = df["fit_score"]
        
        return X, y_class, y_reg

    def train_classifier(self, X: pd.DataFrame, y: pd.Series):
        """Trains a RandomForest classification model for discrete fit tiers."""
        print("Training Random Forest Classifier...")
        
        # In a real scenario, use stratify=y and test_size=0.2
        # Since our mock data is extremely small (3 samples), we use it all to train for demonstration
        is_mock = len(X) <= 10
        
        if is_mock:
            print("Dataset is very small (mock data). Training on all data without splitting.")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        # Evaluation
        y_pred = clf.predict(X_test)
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Feature Importance
        importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\n--- Feature Importance ---")
        print(importance)

        # Save Model
        model_path = os.path.join(self.model_dir, "rf_classifier.pkl")
        joblib.dump(clf, model_path)
        print(f"\nSaved classifier to {model_path}")
        return clf

    def train_regressor(self, X: pd.DataFrame, y: pd.Series):
        """Trains a continuous regression model for explicit fit scoring (0-100)."""
        print("\nTraining Random Forest Regressor...")
        
        is_mock = len(X) <= 10
        if is_mock:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, y_train)
        
        # Evaluation
        y_pred = reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Regression RMSE: {rmse:.2f}")

        # Save Model
        model_path = os.path.join(self.model_dir, "rf_regressor.pkl")
        joblib.dump(reg, model_path)
        print(f"Saved regressor to {model_path}")
        return reg

# ---------------------------------------------------------
# Usage example: Train models on enriched dataset
# ---------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(base_dir, "data/training_dataset_with_features.json")
    model_dir = os.path.join(base_dir, "models/")
    
    trainer = ModelTrainer(data_path, model_dir)
    
    try:
        X, y_class, y_reg = trainer.load_data()
        print(f"Loaded dataset with {len(X)} samples.\n")
        
        # Train Classification Model (predicts 'good', 'medium', 'poor')
        clf = trainer.train_classifier(X, y_class)
        
        # Train Regression Model (predicts 0-100 score)
        reg = trainer.train_regressor(X, y_reg)
        
    except Exception as e:
        print(f"Training failed: {e}")
