"""
Model 1: Spending Prediction
Predicts total monthly spending for next month using Random Forest + XGBoost ensemble
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from .base_model import BaseFinancialModel

class SpendingPredictionModel(BaseFinancialModel):
    """
    Predicts next month's total spending
    Uses ensemble of Random Forest and Gradient Boosting
    """
    
    def __init__(self, rf_estimators=100, gb_estimators=100):
        super().__init__("Spending Prediction Model")
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=gb_estimators,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.rf_weight = 0.4
        self.gb_weight = 0.6
    
    def train(self, X_train, y_train):
        """Train both RF and GB models"""
        print(f"[{self.model_name}] Training...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        self.rf_model.fit(X_scaled, y_train)
        
        # Train Gradient Boosting
        self.gb_model.fit(X_scaled, y_train)
        
        self.is_trained = True
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Make predictions using weighted ensemble"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Weighted ensemble prediction
        predictions = (self.rf_weight * rf_pred) + (self.gb_weight * gb_pred)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred
        }
        
        return metrics
    
    def feature_importance(self, top_n=10):
        """Get top N important features"""
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # Average importance
        avg_importance = (rf_importance + gb_importance) / 2
        
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
