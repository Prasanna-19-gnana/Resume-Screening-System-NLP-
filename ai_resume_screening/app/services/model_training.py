"""
MODEL TRAINING: Train RandomForest and XGBoost models for resume scoring

TODO before running:
- Install xgboost: pip install xgboost
- Models will be saved to ai_resume_screening/models/ml_scorer.pkl (RandomForest)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Train and evaluate ML models for resume scoring"""
    
    def __init__(self, feature_engineer, current_scorer=None):
        """
        Initialize trainer
        
        Args:
            feature_engineer: FeatureEngineer instance
            current_scorer: Current scorer for label generation (optional)
        """
        self.feature_engineer = feature_engineer
        self.current_scorer = current_scorer
        
        # Will be set during training
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        self.feature_names = self.feature_engineer.get_feature_names()
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train RandomForest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Parameters for RandomForestRegressor
        
        Returns:
            results: Dict with metrics and model info
        """
        
        # Default hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }
        params.update(kwargs)
        
        logger.info(f"Training RandomForest with params: {params}")
        
        # Train model
        self.rf_model = RandomForestRegressor(**params)
        self.rf_model.fit(X_train, y_train)
        
        results = {
            'model_name': 'RandomForest',
            'params': params,
            'feature_importances': dict(zip(self.feature_names, self.rf_model.feature_importances_))
        }
        
        # Evaluate on training set
        y_pred_train = self.rf_model.predict(X_train)
        results['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        results['train_r2'] = r2_score(y_train, y_pred_train)
        
        logger.info(f"Train MAE: {results['train_mae']:.4f}")
        logger.info(f"Train RMSE: {results['train_rmse']:.4f}")
        logger.info(f"Train R²: {results['train_r2']:.4f}")
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            y_pred_val = self.rf_model.predict(X_val)
            results['val_mae'] = mean_absolute_error(y_val, y_pred_val)
            results['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
            results['val_r2'] = r2_score(y_val, y_pred_val)
            
            logger.info(f"Val MAE: {results['val_mae']:.4f}")
            logger.info(f"Val RMSE: {results['val_rmse']:.4f}")
            logger.info(f"Val R²: {results['val_r2']:.4f}")
        
        # Feature importance
        importances = sorted(
            results['feature_importances'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        logger.info("Feature Importances:")
        for feat, imp in importances:
            logger.info(f"  {feat}: {imp:.4f}")
        
        return results
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Parameters for XGBRegressor
        
        Returns:
            results: Dict with metrics and model info
        """
        
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available. Install with: pip install xgboost")
            return {'error': 'XGBoost not installed'}
        
        # Default hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 1
        }
        params.update(kwargs)
        
        logger.info(f"Training XGBoost with params: {params}")
        
        # Train model
        eval_set = None
        if X_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set else None,
            verbose=False
        )
        
        results = {
            'model_name': 'XGBoost',
            'params': params,
            'feature_importances': dict(zip(self.feature_names, self.xgb_model.feature_importances_))
        }
        
        # Evaluate on training set
        y_pred_train = self.xgb_model.predict(X_train)
        results['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        results['train_r2'] = r2_score(y_train, y_pred_train)
        
        logger.info(f"Train MAE: {results['train_mae']:.4f}")
        logger.info(f"Train RMSE: {results['train_rmse']:.4f}")
        logger.info(f"Train R²: {results['train_r2']:.4f}")
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            y_pred_val = self.xgb_model.predict(X_val)
            results['val_mae'] = mean_absolute_error(y_val, y_pred_val)
            results['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
            results['val_r2'] = r2_score(y_val, y_pred_val)
            
            logger.info(f"Val MAE: {results['val_mae']:.4f}")
            logger.info(f"Val RMSE: {results['val_rmse']:.4f}")
            logger.info(f"Val R²: {results['val_r2']:.4f}")
        
        # Feature importance
        importances = sorted(
            results['feature_importances'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        logger.info("Feature Importances:")
        for feat, imp in importances:
            logger.info(f"  {feat}: {imp:.4f}")
        
        return results
    
    def save_model(self, model_name: str = 'rf', output_dir: str = None) -> str:
        """
        Save trained model to disk
        
        Args:
            model_name: 'rf' for RandomForest, 'xgb' for XGBoost
            output_dir: Output directory (default: ai_resume_screening/models)
        
        Returns:
            Path to saved model
        """
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "models"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name == 'rf':
            model = self.rf_model
            filename = "ml_scorer_rf.pkl"
        elif model_name == 'xgb':
            model = self.xgb_model
            filename = "ml_scorer_xgb.pkl"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model is None:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def load_model(filepath: str):
        """Load trained model from disk"""
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def save_training_results(self, results: Dict, output_file: str = None) -> None:
        """Save training results to JSON"""
        
        import json
        
        if output_file is None:
            output_file = Path(__file__).parent.parent.parent / "models" / "training_results.json"
        
        # Convert numpy types to native Python types
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (np.floating, np.integer)):
                results_serializable[key] = float(value)
            elif isinstance(value, dict):
                results_serializable[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            else:
                results_serializable[key] = value
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Training results saved to {output_file}")
