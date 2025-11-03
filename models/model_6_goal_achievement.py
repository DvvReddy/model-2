"""
Model 6: Goal Achievement Prediction
Predicts whether user will achieve their financial goal
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from .base_model import BaseFinancialModel

class GoalAchievementModel(BaseFinancialModel):
    """
    Predicts if user will achieve financial goal
    Uses LightGBM for fast classification
    """
    
    def __init__(self, n_estimators=100, max_depth=7):
        super().__init__("Goal Achievement Model")
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train goal achievement model"""
        print(f"[{self.model_name}] Training...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Predict goal achievement"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get goal achievement probability"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def evaluate(self, X_test, y_test):
        """Evaluate goal achievement prediction"""
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
        """Get top N important features for goal achievement"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
