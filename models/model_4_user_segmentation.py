"""
Model 4: User Segmentation
Clusters users into behavioral segments using K-Means + GMM
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .base_model import BaseFinancialModel

class UserSegmentationModel(BaseFinancialModel):
    """
    Segments users into behavioral groups
    Uses K-Means with optional GMM for soft clustering
    """
    
    def __init__(self, n_clusters=5, method='kmeans'):
        super().__init__("User Segmentation Model")
        self.n_clusters = n_clusters
        self.method = method  # 'kmeans' or 'gmm'
        
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            self.model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        self.scaler = StandardScaler()
        self.segment_labels = [f'Segment_{i}' for i in range(n_clusters)]
    
    def train(self, X_train):
        """Train clustering model"""
        print(f"[{self.model_name}] Training {self.n_clusters} segments using {self.method}...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        print(f"[{self.model_name}] Training complete!")
    
    def predict(self, X):
        """Predict segment for users"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'kmeans':
            predictions = self.model.predict(X_scaled)
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get probability of each segment (soft clustering)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'gmm':
            proba = self.model.predict_proba(X_scaled)
        else:
            # For KMeans, convert distances to probabilities
            distances = self.model.transform(X_scaled)
            proba = 1 / (1 + distances)
            proba = proba / proba.sum(axis=1, keepdims=True)
        
        return proba
    
    def evaluate(self, X_test):
        """Evaluate clustering quality"""
        X_scaled = self.scaler.transform(X_test)
        predictions = self.predict(X_test)
        
        silhouette = silhouette_score(X_scaled, predictions)
        davies_bouldin = davies_bouldin_score(X_scaled, predictions)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'predictions': predictions,
            'n_clusters': self.n_clusters
        }
        
        return metrics
    
    def get_segment_profiles(self, X, segment_names=None):
        """Get profile of each segment"""
        predictions = self.predict(X)
        
        if segment_names is None:
            segment_names = self.segment_labels
        
        profiles = {}
        for i in range(self.n_clusters):
            mask = predictions == i
            profiles[segment_names[i]] = X[mask].mean()
        
        return pd.DataFrame(profiles).T
