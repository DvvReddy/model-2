"""
Model 3: Anomaly Detection
Detects unusual spending transactions using Isolation Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from .base_model import BaseFinancialModel

class AnomalyDetectionModel(BaseFinancialModel):
    """
    Detects anomalous transactions
    Uses Isolation Forest for unsupervised anomaly detection
    """
    
    def __init__(self, contamination=0.05):
        super().__init__("Anomaly Detection Model")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.contamination = contamination
    
    def train(self, X_train):
        """Train anomaly detector on normal data"""
        print(f"[{self.model_name}] Training with contamination={self.contamination}...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """
        Predict anomalies
        Returns: 1 for normal, -1 for anomaly
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get anomaly scores (distance from decision boundary)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        
        # Convert to probability-like scores (0-1)
        proba = 1 / (1 + np.exp(scores))
        
        return proba
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate anomaly detection
        
        y_test: 1 for normal, -1 for anomaly (or 0 for normal, 1 for anomaly)
        """
        predictions = self.predict(X_test)
        
        # Convert predictions to 0/1 format for scoring
        pred_binary = (predictions == -1).astype(int)
        y_binary = (y_test == -1).astype(int) if y_test.min() == -1 else y_test
        
        f1 = f1_score(y_binary, pred_binary, zero_division=0)
        precision = precision_score(y_binary, pred_binary, zero_division=0)
        recall = recall_score(y_binary, pred_binary, zero_division=0)
        
        metrics = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': predictions
        }
        
        return metrics
