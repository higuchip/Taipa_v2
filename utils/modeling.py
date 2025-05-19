import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime
import os
from typing import Dict, Tuple, List, Optional
import logging

class SDMModel:
    """Species Distribution Model using Random Forest"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.feature_importance = None
        self.trained_at = None
        self.features = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_pipeline(self, n_estimators: int = 100, max_depth: Optional[int] = None) -> Pipeline:
        """Create scikit-learn pipeline with preprocessing and Random Forest"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        return pipeline
    
    def prepare_data(self, occurrence_data: pd.DataFrame, pseudo_absence_data: pd.DataFrame,
                    env_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Combine occurrence and pseudo-absence data
        occurrence_data['presence'] = 1
        pseudo_absence_data['presence'] = 0
        
        all_data = pd.concat([occurrence_data, pseudo_absence_data], ignore_index=True)
        
        # Extract features and labels
        X = all_data[env_features].values
        y = all_data['presence'].values
        
        self.features = env_features
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2,
              n_estimators: int = 100,
              max_depth: Optional[int] = None) -> Dict:
        """Train the model with standard train-test split"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )
        
        # Create and train pipeline
        self.model = self.create_pipeline(n_estimators, max_depth)
        
        self.logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Feature importance
        rf_model = self.model.named_steps['classifier']
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.trained_at = datetime.now()
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath: str):
        """Save trained model with metadata"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'features': self.features,
            'trained_at': self.trained_at,
            'metadata': getattr(self, 'metadata', {})
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model with metadata"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.metrics = model_data['metrics']
        self.feature_importance = model_data['feature_importance']
        self.features = model_data['features']
        self.trained_at = model_data['trained_at']
        self.metadata = model_data.get('metadata', {})
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance dataframe"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance
    
    def get_metrics(self) -> Dict:
        """Get model performance metrics"""
        return self.metrics