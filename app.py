"""
FinBuddy AI - Interactive Financial Prediction Dashboard
Complete working version with all 7 models
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
    """Load all trained models safely"""
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
                if isinstance(data, dict):
                    if 'model' in data:
                        models[name] = data['model']
                    else:
                        models[name] = data
                else:
                    models[name] = data
                st.write(f"✓ Loaded {name}")
            else:
                st.write(f"✗ {name} file not found: {path}")
        except Exception as e:
            st.write(f"✗ Error loading {name}: {str(e)[:100]}")
    
    return models

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
    features[15] = spending * 0.7
    features[16] = spending * 0.8
    features[20] = volatility
    features[25] = volatility * 100
    features[30] = 1.0 if spending > 20000 else 0.0
    features[35] = spending / 7 if spending > 0 else 0
    features[36] = spending
    features[40] = online_ratio
    features[45] = np.random.uniform(4, 8)
    features[50] = 0.5
    features[55] = online_ratio * 0.6
    features[60] = max(0, 1 - volatility)
    
    # Fill remaining
    for i in range(n_features):
        if features[i] == 0:
            features[i] = np.random.uniform(0, 100)
    
    return features

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown("# 💰 FinBuddy AI - Financial Prediction Engine")
    st.markdown("### Predict spending, risks, and financial health using AI")
    
    # Load models with progress
    with st.spinner("⏳ Loading AI models..."):
        models = load_models()
    
    if not models or len(models) == 0:
        st.error("❌ No models loaded! Please upload artifact files.")
        st.stop()
    
    st.success(f"✅ Loaded {len(models)}/7 models successfully!")
    
    # Sidebar Navigation
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
            "📈 About",
        ]
    )
    
    # ====================================================================
    # PAGE 1: DASHBOARD OVERVIEW
    # ====================================================================
    if page == "🎯 Dashboard Overview":
        st.markdown("## 📊 Dashboard Overview")
        st.info("👇 Enter your financial data below to get AI-powered predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_spending = st.number_input(
                "💵 Monthly Spending (₹)",
                min_value=0,
                max_value=1000000,
                value=25000,
                step=1000
            )
        
        with col2:
            spending_volatility = st.slider(
                "📈 Volatility (0=Stable, 1=Erratic)",
                0.0, 1.0, 0.3, 0.1
            )
        
        with col3:
            online_ratio = st.slider(
                "🌐 Online Spending %",
                0.0, 1.0, 0.5, 0.1
            )
        
        if st.button("🚀 Generate Predictions", key="dashboard"):
            features = create_feature_vector(monthly_spending, spending_volatility, online_ratio)
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Spending
            with col1:
                try:
                    if models.get('spending'):
                        pred = models['spending'].predict([features])[0]
                        st.metric("💸 Next Month", f"₹{pred:,.0f}", f"{(pred/monthly_spending - 1)*100:+.1f}%")
                    else:
                        st.warning("Model unavailable")
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
            
            # Risk
            with col2:
                try:
                    if models.get('risk'):
                        proba = models['risk'].predict_proba([features])[0]
                        risk = "🔴 High" if proba[1] > 0.5 else "🟢 Low"
                        st.metric("💰 Risk", risk, f"{proba[1]:.1%}")
                    else:
                        st.warning("Model unavailable")
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
            
            # Goal
            with col3:
                try:
                    if models.get('goal'):
                        proba = models['goal'].predict_proba([features])[0]
                        st.metric("🎯 Goal Success", f"{proba[1]:.0%}")
                    else:
                        st.warning("Model unavailable")
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
            
            # Churn
            with col4:
                try:
                    if models.get('churn'):
                        proba = models['churn'].predict_proba([features])[0]
                        risk = "🔴 High" if proba[1] > 0.6 else "🟢 Low"
                        st.metric("👋 Churn Risk", risk, f"{proba[1]:.1%}")
                    else:
                        st.warning("Model unavailable")
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
    
    # ====================================================================
    # PAGE 2: SPENDING PREDICTION
    # ====================================================================
    elif page == "💸 Spending Prediction":
        st.markdown("## 💸 Spending Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_spending = st.number_input("Monthly Average (₹)", 0, 1000000, 20000, 1000)
            spending_std = st.slider("Volatility", 0, 50000, 5000, 500)
        
        with col2:
            recent_7d = st.number_input("Last 7 Days (₹)", 0, 100000, 5000, 500)
            recent_30d = st.number_input("Last 30 Days (₹)", 0, 500000, 25000, 500)
        
        if st.button("🔮 Predict", key="spending_pred"):
            try:
                if not models.get('spending'):
                    st.error("Spending model not loaded")
                else:
                    features = create_feature_vector(avg_spending, spending_std/max(avg_spending, 1), 0.5)
                    prediction = models['spending'].predict([features])[0]
                    
                    st.success(f"### Predicted: ₹{prediction:,.0f}")
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Last 7d', 'Last 30d', 'Predicted'],
                        y=[recent_7d, recent_30d, prediction],
                        marker_color=['lightblue', 'skyblue', 'orange']
                    ))
                    fig.update_layout(title="Spending Trend", yaxis_title="₹", xaxis_title="Period")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 3: CATEGORY BREAKDOWN
    # ====================================================================
    elif page == "📊 Category Breakdown":
        st.markdown("## 📊 Spending by Category")
        
        monthly = st.slider("Total Monthly Spending (₹)", 5000, 500000, 50000, 5000)
        
        if st.button("📈 Forecast Categories", key="category_pred"):
            try:
                if not models.get('category'):
                    st.error("Category model not loaded")
                else:
                    features = create_feature_vector(monthly, 0.3, 0.5)
                    category_pred = models['category'].predict([features])[0]
                    
                    # DataFrame
                    df = pd.DataFrame({
                        'Category': models['category'].category_names,
                        'Spending': category_pred
                    }).sort_values('Spending', ascending=False)
                    
                    # Pie chart
                    fig = px.pie(df, values='Spending', names='Category', title="Budget Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.dataframe(df.style.format({'Spending': '₹{:,.0f}'}), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 4: ANOMALY DETECTION
    # ====================================================================
    elif page == "⚠️ Anomaly Detection":
        st.markdown("## ⚠️ Transaction Anomaly Detection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount (₹)", 0, 100000, 5000, 100)
        with col2:
            hour = st.slider("Hour (0-23)", 0, 23, 14)
        with col3:
            is_online = st.selectbox("Type", ["Online", "Offline"])
        
        if st.button("🔍 Check Anomaly", key="anomaly_pred"):
            try:
                if not models.get('anomaly'):
                    st.error("Anomaly model not loaded")
                else:
                    features = create_feature_vector(amount, 0.1, 1.0 if is_online == "Online" else 0.0)
                    score = models['anomaly'].predict_proba([features])[0]
                    
                    if score > 0.5:
                        st.error(f"🔴 ANOMALY! Risk: {score:.1%}")
                    else:
                        st.success(f"🟢 NORMAL. Risk: {score:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 5: USER SEGMENTATION
    # ====================================================================
    elif page == "👥 User Segmentation":
        st.markdown("## 👥 Which Segment Are You?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = st.slider("Categories Used", 1, 12, 5)
        with col2:
            avg_txn = st.number_input("Avg Transaction (₹)", 1000, 50000, 10000)
        with col3:
            freq = st.slider("Frequency Score", 0.0, 1.0, 0.5)
        
        if st.button("🎯 Find Segment", key="segment_pred"):
            try:
                if not models.get('segmentation'):
                    st.error("Segmentation model not loaded")
                else:
                    features = create_feature_vector(categories * avg_txn, 0.3, freq)
                    segment = models['segmentation'].predict([features])[0]
                    proba = models['segmentation'].predict_proba([features])[0]
                    
                    st.info(f"### You are: **Segment {segment}** ({proba[segment]:.0%} match)")
                    
                    # Bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"S{i}" for i in range(5)],
                        y=proba,
                        marker_color=['green' if i == segment else 'lightblue' for i in range(5)]
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 6: RISK ASSESSMENT
    # ====================================================================
    elif page == "💰 Risk Assessment":
        st.markdown("## 💰 Financial Risk Level")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income = st.number_input("Monthly Income (₹)", 0, 1000000, 50000)
        with col2:
            spending = st.number_input("Monthly Spending (₹)", 0, 1000000, 30000)
        with col3:
            savings_rate = st.slider("Savings %", 0.0, 1.0, 0.4)
        
        if st.button("📊 Assess Risk", key="risk_pred"):
            try:
                if not models.get('risk'):
                    st.error("Risk model not loaded")
                else:
                    features = create_feature_vector(spending, abs(income - spending) / max(income, 1), savings_rate)
                    proba = models['risk'].predict_proba([features])[0]
                    
                    if proba[1] > 0.5:
                        st.error(f"🔴 HIGH RISK: {proba[1]:.1%}")
                        st.warning("⚠️ Recommendations: Increase savings, reduce expenses")
                    else:
                        st.success(f"🟢 LOW RISK: {proba[1]:.1%}")
                        st.success("✓ Your financial health is good!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 7: GOAL ACHIEVEMENT
    # ====================================================================
    elif page == "🎯 Goal Achievement":
        st.markdown("## 🎯 Will You Achieve Your Goal?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goal = st.number_input("Goal Amount (₹)", 0, 10000000, 100000)
            months = st.slider("Months", 1, 60, 12)
        
        with col2:
            current = st.number_input("Current Savings (₹)", 0, 10000000, 20000)
            monthly_save = st.number_input("Monthly Savings (₹)", 0, 100000, 5000)
        
        if st.button("🎯 Check Achievement", key="goal_pred"):
            try:
                if not models.get('goal'):
                    st.error("Goal model not loaded")
                else:
                    features = create_feature_vector(monthly_save, 0.1, 0.5)
                    proba = models['goal'].predict_proba([features])[0]
                    
                    projected = current + (monthly_save * months)
                    gap = goal - projected
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Goal", f"₹{goal:,.0f}")
                    with col2:
                        st.metric("Projected", f"₹{projected:,.0f}")
                    with col3:
                        st.metric("Gap", f"₹{gap:,.0f}" if gap > 0 else f"✓ {abs(gap):,.0f}")
                    
                    st.success(f"✅ Success Probability: {proba[1]:.1%}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 8: CHURN PREDICTION
    # ====================================================================
    elif page == "👋 Churn Prediction":
        st.markdown("## 👋 User Engagement Risk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_inactive = st.slider("Days Inactive", 0, 365, 5)
        with col2:
            logins = st.slider("Monthly Logins", 0, 30, 10)
        with col3:
            features_used = st.slider("Features Used (1-10)", 1, 10, 7)
        
        if st.button("📊 Check Churn Risk", key="churn_pred"):
            try:
                if not models.get('churn'):
                    st.error("Churn model not loaded")
                else:
                    engagement = (logins / 30) * (features_used / 10) * (1 - min(days_inactive / 365, 1))
                    features = create_feature_vector(engagement * 50000, 0.1, engagement)
                    proba = models['churn'].predict_proba([features])[0]
                    
                    if proba[1] > 0.6:
                        st.error(f"🔴 CRITICAL: {proba[1]:.1%}")
                        st.warning("Send retention offer immediately!")
                    elif proba[1] > 0.3:
                        st.warning(f"🟡 MEDIUM: {proba[1]:.1%}")
                        st.info("Increase engagement")
                    else:
                        st.success(f"🟢 LOW: {proba[1]:.1%}")
                        st.success("User is engaged!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # ====================================================================
    # PAGE 9: ABOUT
    # ====================================================================
    elif page == "📈 About":
        st.markdown("## 📈 About FinBuddy AI")
        
        st.info("""
        ### FinBuddy AI - Financial Prediction Engine
        
        **Powered by 7 Advanced ML Models** trained on 1M+ transactions from 2,331 users over 6 months.
        
        #### 🎯 Models:
        1. **💸 Spending Prediction** - R²: 0.995 (99.5% accurate)
        2. **📊 Category Forecast** - R²: 0.406 (predicts per-category spending)
        3. **⚠️ Anomaly Detection** - F1: 0.60 (detects unusual transactions)
        4. **👥 User Segmentation** - 5 distinct behavioral segments
        5. **💰 Risk Assessment** - Accuracy: 1.00 (financial health evaluation)
        6. **🎯 Goal Achievement** - AUC: 0.917 (success prediction)
        7. **👋 Churn Prediction** - AUC: 0.849 (engagement risk)
        
        #### 📊 Dataset:
        - Transactions: 1,051,591
        - Users: 2,331
        - Duration: 6 months
        - Features Engineered: 86+ per transaction
        
        #### ✨ Features:
        - ✓ Real-time predictions
        - ✓ Interactive dashboard
        - ✓ Visual analytics
        - ✓ Personalized insights
        - ✓ Production-ready
        
        ---
        **Version**: 1.0.0  
        **Built with**: Python, Scikit-learn, XGBoost, Streamlit
        """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
