"""
Model 2: Category-Specific Spending Forecast
Multi-output regression predicting spending for all categories
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from .base_model import BaseFinancialModel

class CategoryForecastModel(BaseFinancialModel):
    """
    Predicts spending per category for next month
    Handles multiple output targets (one per category)
    """
    
    def __init__(self, n_estimators=100):
        super().__init__("Category Forecast Model")
        base_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model = MultiOutputRegressor(base_model)
        self.scaler = StandardScaler()
        self.category_names = None
    
    def train(self, X_train, y_train_categories):
        """
        Train multi-output model
        
        y_train_categories: DataFrame with category columns
        """
        print(f"[{self.model_name}] Training on {len(y_train_categories.columns)} categories...")
        
        self.feature_names = X_train.columns.tolist()
        self.category_names = y_train_categories.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train_categories)
        
        self.is_trained = True
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Predict spending for each category"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return pd.DataFrame(predictions, columns=self.category_names)
    
    def evaluate(self, X_test, y_test_categories):
        """Evaluate per-category performance"""
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test_categories, y_pred)
        r2 = r2_score(y_test_categories, y_pred, multioutput='raw_values')
        
        metrics = {
            'mae': mae,
            'r2_per_category': dict(zip(self.category_names, r2)),
            'avg_r2': np.mean(r2),
            'predictions': y_pred
        }
        
        return metrics
