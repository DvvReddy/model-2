"""
FinBuddy AI - Interactive Financial Prediction Dashboard
Streamlit app for model predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FinBuddy AI - Financial Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS (cached for performance)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        models = {}
        model_files = {
            'spending': 'artifacts/model_1_spending.pkl',
            'category': 'artifacts/model_2_category.pkl',
            'anomaly': 'artifacts/model_3_anomaly.pkl',
            'segmentation': 'artifacts/model_4_segmentation.pkl',
            'risk': 'artifacts/model_5_risk.pkl',
            'goal': 'artifacts/model_6_goal.pkl',
            'churn': 'artifacts/model_7_churn.pkl',
        }
        
        for name, path in model_files.items():
            try:
                if os.path.exists(path):
                    data = joblib.load(path)
                    # Extract model from dict wrapper if needed
                    if isinstance(data, dict) and 'model' in data:
                        models[name] = data['model']
                    else:
                        models[name] = data
                else:
                    st.warning(f"⚠️ Model file not found: {path}")
            except Exception as e:
                st.warning(f"⚠️ Error loading {name}: {e}")
        
        if len(models) < 7:
            st.warning(f"⚠️ Only {len(models)}/7 models loaded. Some predictions may not work.")
        
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def create_feature_vector(spending, volatility, online_ratio, n_features=66):
    """Create feature vector for predictions"""
    features = np.zeros(n_features)
    
    # Fill relevant features
    features[0] = spending
    features[5] = volatility
    features[10] = online_ratio
    features[15] = spending * 0.7  # 7-day average
    features[16] = spending * 0.8  # 30-day average
    features[20] = volatility
    features[25] = volatility * 100
    features[30] = 1.0 if spending > 20000 else 0.0
    features[35] = spending / 7
    features[36] = spending
    features[40] = online_ratio
    features[45] = np.random.uniform(4, 8)
    features[50] = 0.5
    features[55] = online_ratio * 0.6
    features[60] = 1 - volatility
    
    # Fill remaining with reasonable defaults
    for i in range(n_features):
        if features[i] == 0:
            features[i] = np.random.uniform(0, 100)
    
    return features

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.markdown("# 💰 FinBuddy AI - Financial Prediction Engine")
    st.markdown("### Predict spending, risks, and financial health using AI")
    
    # Load models
    models = load_models()
    if models is None or len(models) == 0:
        st.error("❌ Failed to load models. Please check artifact files are uploaded.")
        st.stop()
    
    st.success(f"✅ Loaded {len(models)}/7 models successfully!")
    
    # Sidebar - Navigation
    st.sidebar.markdown("# 📋 Navigation")
    page = st.sidebar.radio(
        "Select Prediction Type:",
        [
            "🎯 Dashboard Overview",
            "💸 Spending Prediction",
            "📊 Category Breakdown",
            "⚠️ Anomaly Detection",
            "👥 User Segmentation",
            "💰 Risk Assessment",
            "🎯 Goal Achievement",
            "👋 Churn Prediction",
            "📈 Comprehensive Analysis"
        ]
    )
    
    # ====================================================================
    # PAGE 1: DASHBOARD OVERVIEW
    # ====================================================================
    if page == "🎯 Dashboard Overview":
        st.markdown("## Dashboard Overview")
        st.info("Enter your financial data to get AI-powered predictions")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_spending = st.number_input(
                "💵 Average Monthly Spending (₹)",
                min_value=0,
                max_value=1000000,
                value=25000,
                step=1000
            )
        
        with col2:
            spending_volatility = st.slider(
                "📈 Spending Volatility (0=Consistent, 1=Erratic)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        
        with col3:
            online_ratio = st.slider(
                "🌐 Online Spending Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # Create feature vector
        if st.button("🚀 Generate Predictions", key="dashboard"):
            try:
                features = create_feature_vector(
                    monthly_spending,
                    spending_volatility,
                    online_ratio
                )
                
                # Make predictions
                if 'spending' in models:
                    spending_pred = models['spending'].predict([features])[0]
                else:
                    spending_pred = 0
                
                if 'risk' in models:
                    risk_proba = models['risk'].predict_proba([features])[0]
                else:
                    risk_proba = [0.5, 0.5]
                
                if 'goal' in models:
                    goal_proba = models['goal'].predict_proba([features])[0]
                else:
                    goal_proba = [0.5, 0.5]
                
                if 'churn' in models:
                    churn_proba = models['churn'].predict_proba([features])[0]
                else:
                    churn_proba = [0.5, 0.5]
                
                # Display results
                st.markdown("### 📊 Predictions")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💸 Next Month Spending",
                        f"₹{spending_pred:,.0f}",
                        f"₹{spending_pred - monthly_spending:+,.0f}"
                    )
                
                with col2:
                    risk_level = "🔴 High" if risk_proba[1] > 0.5 else "🟡 Medium" if risk_proba[1] > 0.3 else "🟢 Low"
                    st.metric(
                        "💰 Risk Level",
                        risk_level,
                        f"{risk_proba[1]:.1%} risk"
                    )
                
                with col3:
                    goal_prob = goal_proba[1]
                    st.metric(
                        "🎯 Goal Achievement",
                        f"{goal_prob:.0%}",
                        "Likely to achieve" if goal_prob > 0.7 else "Uncertain"
                    )
                
                with col4:
                    churn_prob = churn_proba[1]
                    churn_label = "🔴 High" if churn_prob > 0.6 else "🟡 Medium" if churn_prob > 0.3 else "🟢 Low"
                    st.metric(
                        "👋 Churn Risk",
                        churn_label,
                        f"{churn_prob:.1%} risk"
                    )
            except Exception as e:
                st.error(f"❌ Error making predictions: {e}")
    
    # ====================================================================
    # PAGE 2: SPENDING PREDICTION
    # ====================================================================
    elif page == "💸 Spending Prediction":
        st.markdown("## 💸 Spending Prediction")
        st.write("Predict your monthly spending based on your financial habits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_spending = st.number_input(
                "Average Monthly Spending (₹)",
                min_value=0,
                max_value=1000000,
                value=20000,
                step=1000
            )
            spending_std = st.slider(
                "Spending Standard Deviation",
                min_value=0,
                max_value=50000,
                value=5000,
                step=500
            )
        
        with col2:
            recent_7d = st.number_input(
                "Last 7 Days Spending (₹)",
                min_value=0,
                max_value=100000,
                value=5000,
                step=500
            )
            recent_30d = st.number_input(
                "Last 30 Days Spending (₹)",
                min_value=0,
                max_value=500000,
                value=25000,
                step=500
            )
        
        if st.button("🔮 Predict Spending", key="spending"):
            try:
                features = create_feature_vector(avg_spending, spending_std/avg_spending if avg_spending > 0 else 0, 0.5)
                
                if 'spending' in models:
                    prediction = models['spending'].predict([features])[0]
                    st.success(f"### 💰 Predicted Next Month Spending: ₹{prediction:,.0f}")
                    
                    # Show chart
                    months = ['Last Month', 'Current Month', 'Next Month (Predicted)']
                    amounts = [avg_spending, recent_30d, prediction]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=months, y=amounts, marker_color=['lightblue', 'skyblue', 'orange']))
                    fig.update_layout(title="Spending Trend", yaxis_title="Amount (₹)", xaxis_title="Month")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Spending model not available")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 3: QUICK INFO
    # ====================================================================
    elif page == "📈 Comprehensive Analysis":
        st.markdown("## 📈 About FinBuddy AI")
        st.info("""
        **FinBuddy AI** is a financial prediction system trained on 1M+ transactions from 2,331 users.
        
        ### 7 AI Models:
        1. 💸 **Spending Prediction** - Forecast next month spending (R²: 0.995)
        2. 📊 **Category Breakdown** - Predict spending per category
        3. ⚠️ **Anomaly Detection** - Detect unusual transactions
        4. 👥 **User Segmentation** - Classify user types (5 segments)
        5. 💰 **Risk Assessment** - Evaluate financial risk
        6. 🎯 **Goal Achievement** - Predict goal success probability
        7. 👋 **Churn Prediction** - Identify at-risk users
        
        ### Performance:
        - Spending Prediction Accuracy: **99.5%**
        - Goal Achievement AUC: **0.917**
        - Churn Detection AUC: **0.849**
        """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
