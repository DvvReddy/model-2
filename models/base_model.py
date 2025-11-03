"""
Base Model Class
Provides common interface for all models
"""

from abc import ABC, abstractmethod
import joblib

class BaseFinancialModel(ABC):
    """Abstract base class for all financial prediction models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train=None):
        """Train the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions - must be implemented by subclasses"""
        pass
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_name': self.model_name
        }, path)
        print(f"✓ Model saved: {path}")
    
    def load(self, path):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = True
        print(f"✓ Model loaded: {path}")
    
    def __repr__(self):
        return f"{self.model_name} (Trained: {self.is_trained})"
