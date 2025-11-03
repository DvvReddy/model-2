"""
Model 7: Churn Prediction
Predicts user churn/disengagement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from .base_model import BaseFinancialModel

class ChurnPredictionModel(BaseFinancialModel):
    """
    Predicts user churn/disengagement
    Uses Random Forest with class weighting for imbalanced data
    """
    
    def __init__(self, n_estimators=100):
        super().__init__("Churn Prediction Model")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            class_weight='balanced',  # Handle imbalanced data
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train churn prediction model"""
        print(f"[{self.model_name}] Training...")
        print(f"  Churn distribution: {y_train.value_counts().to_dict()}")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Predict churn"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get churn probability"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def evaluate(self, X_test, y_test):
        """Evaluate churn prediction"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return metrics

    
    def feature_importance(self, top_n=10):
        """Get top N important features for churn"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
