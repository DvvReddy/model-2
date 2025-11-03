"""
Model 5: Risk Assessment
Classifies users into risk levels (Low/Medium/High)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from .base_model import BaseFinancialModel

class RiskAssessmentModel(BaseFinancialModel):
    """
    Assesses financial risk level for users
    Uses XGBoost for classification
    """
    
    def __init__(self, max_depth=6, learning_rate=0.1):
        super().__init__("Risk Assessment Model")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss'
        )
        self.scaler = StandardScaler()
        self.class_names = ['Low Risk', 'Medium Risk', 'High Risk']
    
    def train(self, X_train, y_train):
        """Train risk assessment model"""
        print(f"[{self.model_name}] Training...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_scaled, y_train, verbose=0)
        self.is_trained = True
        
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Predict risk level"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get risk probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def evaluate(self, X_test, y_test):
        """Evaluate risk assessment"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Handle case where only one class is present in y_test
        try:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1], multi_class='ovr')
        except ValueError:
            roc_auc = 0.0  # Set to 0 if only one class present
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return metrics

    
    def feature_importance(self, top_n=10):
        """Get top N important features for risk"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
