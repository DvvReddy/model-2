"""
FinBuddy AI - Interactive Financial Prediction Dashboard
Streamlit app for model predictions - FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FinBuddy AI - Financial Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1f77e6;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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
                data = joblib.load(path)
                # Extract actual model from dict if needed
                if isinstance(data, dict) and 'model' in data:
                    models[name] = data['model']
                else:
                    models[name] = data
            except FileNotFoundError:
                st.error(f"❌ Model file not found: {path}")
                return None
            except Exception as e:
                st.error(f"❌ Error loading {name} model: {e}")
                return None
        
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.markdown("# 💰 FinBuddy AI - Financial Prediction Engine")
    st.markdown("Predict spending, risks, and financial health using AI")
    
    # Load models
    models = load_models()
    if models is None:
        st.error("❌ Failed to load models. Please check artifact files.")
        st.info("📝 Make sure all model files are in the 'artifacts/' directory")
        st.stop()  # ✅ FIXED: Properly indented

    
    # Sidebar - Navigation
    st.sidebar.markdown("# 📋 Navigation")
    page = st.sidebar.radio(
        "Select Prediction Type:",
        [
            "Dashboard Overview",
            "💳 Spending Prediction",
            "📊 Category Breakdown",
            "🚨 Anomaly Detection",
            "👥 User Segmentation",
            "⚠️ Risk Assessment",
            "🎯 Goal Achievement",
            "📉 Churn Prediction",
            "🔍 Comprehensive Analysis"
        ]
    )
    
    # ====================================================================
    # PAGE 1: DASHBOARD OVERVIEW
    # ====================================================================
    if page == "Dashboard Overview":
        st.markdown("## 📊 Dashboard Overview")
        st.info("Enter your financial data to get AI-powered predictions")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_spending = st.number_input(
                "Average Monthly Spending (₹)",
                min_value=0,
                max_value=1000000,
                value=25000,
                step=1000
            )
        
        with col2:
            spending_volatility = st.slider(
                "Spending Volatility (0=Consistent, 1=Erratic)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        
        with col3:
            online_ratio = st.slider(
                "💻 Online Spending Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # Create feature vector
        if st.button("🚀 Generate Predictions", key="dashboard"):
            features = create_feature_vector(
                monthly_spending,
                spending_volatility,
                online_ratio
            )
            
            # Make predictions with error handling
            try:
                spending_pred = models['spending'].predict([features])[0]
                risk_proba = models['risk'].predict_proba([features])[0]
                goal_proba = models['goal'].predict_proba([features])[0]
                churn_proba = models['churn'].predict_proba([features])[0]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return
            
            # Display results
            st.markdown("### 📈 Predictions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Next Month Spending",
                    f"₹{spending_pred:,.0f}",
                    f"₹{spending_pred - monthly_spending:+,.0f}"
                )
            
            with col2:
                risk_level = "🔴 High" if risk_proba[1] > 0.5 else "🟡 Medium" if risk_proba[1] > 0.3 else "🟢 Low"
                st.metric(
                    "Risk Level",
                    risk_level,
                    f"{risk_proba[1]:.1%} risk"
                )
            
            with col3:
                goal_prob = goal_proba[1]
                st.metric(
                    "Goal Achievement",
                    f"{goal_prob:.0%}",
                    "✅ Likely to achieve" if goal_prob > 0.7 else "⚠️ Uncertain"
                )
            
            with col4:
                churn_prob = churn_proba[1]
                churn_label = "🔴 High" if churn_prob > 0.6 else "🟡 Medium" if churn_prob > 0.3 else "🟢 Low"
                st.metric(
                    "Churn Risk",
                    churn_label,
                    f"{churn_prob:.1%} risk"
                )
    
    # ====================================================================
    # PAGE 2: SPENDING PREDICTION
    # ====================================================================
    elif page == "💳 Spending Prediction":
        st.markdown("## 💳 Spending Prediction")
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
            features = create_feature_vector(
                avg_spending, 
                spending_std/avg_spending if avg_spending > 0 else 0, 
                0.5
            )
            try:
                prediction = models['spending'].predict([features])[0]
                
                st.success(f"💰 Predicted Next Month Spending: ₹{prediction:,.0f}")
                
                # Show chart
                months = ['Last Month', 'Current Month', 'Next Month (Predicted)']
                amounts = [avg_spending, recent_30d, prediction]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=months, 
                    y=amounts, 
                    marker_color=['lightblue', 'skyblue', 'orange']
                ))
                fig.update_layout(
                    title="Spending Trend", 
                    yaxis_title="Amount (₹)", 
                    xaxis_title="Month"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # ====================================================================
    # PAGE 3: CATEGORY BREAKDOWN
    # ====================================================================
    elif page == "📊 Category Breakdown":
        st.markdown("## 📊 Category-wise Spending Forecast")
        
        monthly_spending = st.slider(
            "Total Monthly Spending (₹)",
            min_value=5000,
            max_value=500000,
            value=50000,
            step=5000
        )
        
        if st.button("📈 Forecast Categories", key="category"):
            features = create_feature_vector(monthly_spending, 0.3, 0.5)
            try:
                category_pred = models['category'].predict([features])[0]
                
                # Check if category_names exists, otherwise use generic names
                if hasattr(models['category'], 'category_names'):
                    category_names = models['category'].category_names
                else:
                    category_names = [f"Category {i+1}" for i in range(len(category_pred))]
                
                # Create dataframe
                df_cat = pd.DataFrame({
                    'Category': category_names,
                    'Spending': category_pred
                }).sort_values('Spending', ascending=False)
                
                # Pie chart
                fig = px.pie(
                    df_cat, 
                    values='Spending', 
                    names='Category', 
                    title="Category Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    df_cat.style.format({'Spending': '₹{:,.0f}'}), 
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error forecasting categories: {e}")
    
    # ====================================================================
    # PAGE 4: ANOMALY DETECTION
    # ====================================================================
    elif page == "🚨 Anomaly Detection":
        st.markdown("## 🚨 Detect Unusual Transactions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transaction_amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=0,
                max_value=100000,
                value=5000,
                step=100
            )
        
        with col2:
            hour = st.slider(
                "Hour of Transaction (0-23)",
                min_value=0,
                max_value=23,
                value=14
            )
        
        with col3:
            is_online = st.selectbox("Transaction Type", ["Online", "Offline"])
        
        if st.button("🔍 Check for Anomaly", key="anomaly"):
            features = create_feature_vector(
                transaction_amount, 
                0.1, 
                1.0 if is_online == "Online" else 0.0
            )
            try:
                # ✅ FIXED: Get the anomaly probability correctly
                anomaly_proba = models['anomaly'].predict_proba([features])[0]
                anomaly_score = anomaly_proba[1] if len(anomaly_proba) > 1 else anomaly_proba[0]
                
                if anomaly_score > 0.5:
                    st.warning(f"⚠️ ANOMALY DETECTED! Risk Score: {anomaly_score:.1%}")
                else:
                    st.success(f"✅ NORMAL TRANSACTION. Risk Score: {anomaly_score:.1%}")
            except Exception as e:
                st.error(f"Error detecting anomaly: {e}")
    
    # ====================================================================
    # PAGE 5: USER SEGMENTATION
    # ====================================================================
    elif page == "👥 User Segmentation":
        st.markdown("## 👥 User Segment Classification")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_count = st.slider("Number of Categories Used", 1, 12, 5)
        
        with col2:
            avg_transaction = st.number_input("Average Transaction (₹)", 1000, 50000, 10000)
        
        with col3:
            freq_score = st.slider("Transaction Frequency", 0.0, 1.0, 0.5)
        
        if st.button("🔍 Find Segment", key="segment"):
            features = create_feature_vector(
                category_count * avg_transaction, 
                0.3, 
                freq_score
            )
            try:
                segment = models['segmentation'].predict([features])[0]
                
                # Check if predict_proba exists
                if hasattr(models['segmentation'], 'predict_proba'):
                    segment_proba = models['segmentation'].predict_proba([features])[0]
                    n_segments = len(segment_proba)
                else:
                    n_segments = 5
                    segment_proba = [0.2] * n_segments
                    segment_proba[segment] = 0.8
                
                st.info(f"### 🎯 You belong to: **Segment {segment}**")
                
                # Show segment distribution
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Segment {i}" for i in range(n_segments)],
                    y=segment_proba,
                    marker_color=['green' if i == segment else 'lightblue' for i in range(n_segments)]
                ))
                fig.update_layout(title="Segment Probability", yaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error finding segment: {e}")
    
    # ====================================================================
    # PAGE 6: RISK ASSESSMENT
    # ====================================================================
    elif page == "⚠️ Risk Assessment":
        st.markdown("## ⚠️ Financial Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_income = st.number_input(
                "Monthly Income (₹)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000
            )
        
        with col2:
            monthly_spending = st.number_input(
                "Monthly Spending (₹)",
                min_value=0,
                max_value=1000000,
                value=30000,
                step=5000
            )
        
        with col3:
            savings_rate = st.slider(
                "Savings Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.4
            )
        
        if st.button("🔍 Assess Risk", key="risk"):
            features = create_feature_vector(
                monthly_spending, 
                abs(monthly_income - monthly_spending) / monthly_income if monthly_income > 0 else 0, 
                savings_rate
            )
            try:
                risk_proba = models['risk'].predict_proba([features])[0]
                
                risk_level = 0 if risk_proba[0] > 0.5 else 1
                risk_label = ["🟢 Low Risk", "🔴 High Risk"][risk_level]
                
                st.success(f"### {risk_label}")
                st.write(f"Risk Score: {risk_proba[risk_level]:.1%}")
                
                if risk_level == 1:
                    st.warning("⚠️ Recommendations: Increase savings, reduce unnecessary expenses")
                else:
                    st.success("✅ Your financial health looks good!")
            except Exception as e:
                st.error(f"Error assessing risk: {e}")
    
    # ====================================================================
    # PAGE 7: GOAL ACHIEVEMENT
    # ====================================================================
    elif page == "🎯 Goal Achievement":
        st.markdown("## 🎯 Will You Achieve Your Financial Goal?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goal_amount = st.number_input(
                "Financial Goal Amount (₹)",
                min_value=0,
                max_value=10000000,
                value=100000,
                step=10000
            )
            timeframe_months = st.slider(
                "Timeframe (months)",
                min_value=1,
                max_value=60,
                value=12
            )
        
        with col2:
            current_savings = st.number_input(
                "Current Savings (₹)",
                min_value=0,
                max_value=10000000,
                value=20000,
                step=5000
            )
            monthly_savings = st.number_input(
                "Monthly Savings Capacity (₹)",
                min_value=0,
                max_value=100000,
                value=5000,
                step=500
            )
        
        if st.button("🎯 Check Goal Achievement", key="goal"):
            features = create_feature_vector(monthly_savings, 0.1, 0.5)
            try:
                goal_proba = models['goal'].predict_proba([features])[0]
                
                projected_savings = current_savings + (monthly_savings * timeframe_months)
                achievement_prob = goal_proba[1]
                
                st.metric("Achievement Probability", f"{achievement_prob:.1%}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Goal Amount", f"₹{goal_amount:,.0f}")
                with col2:
                    st.metric("Projected Savings", f"₹{projected_savings:,.0f}")
                with col3:
                    shortfall = goal_amount - projected_savings
                    st.metric(
                        "Gap", 
                        f"₹{shortfall:,.0f}" if shortfall > 0 else f"₹{abs(shortfall):,.0f} Surplus"
                    )
                
                if achievement_prob > 0.7:
                    st.success(f"✅ You will likely achieve your goal!")
                else:
                    st.warning(f"⚠️ You may not achieve your goal. Increase monthly savings.")
            except Exception as e:
                st.error(f"Error checking goal: {e}")
    
    # ====================================================================
    # PAGE 8: CHURN PREDICTION
    # ====================================================================
    elif page == "📉 Churn Prediction":
        st.markdown("## 📉 User Engagement & Churn Risk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_since_last = st.slider(
                "Days Since Last Activity",
                min_value=0,
                max_value=365,
                value=5
            )
            login_frequency = st.slider(
                "Monthly Logins",
                min_value=0,
                max_value=30,
                value=10
            )
        
        with col2:
            features_used = st.slider(
                "Features Used (out of 10)",
                min_value=0,
                max_value=10,
                value=7
            )
        
        with col3:
            goal_progress = st.slider(
                "Goal Progress (%)",
                min_value=0,
                max_value=100,
                value=50
            )
        
        if st.button("📊 Predict Churn Risk", key="churn"):
            # Create engagement score
            engagement_score = (login_frequency / 30) * (features_used / 10) * (100 - days_since_last / 365 * 100) / 100
            
            features = create_feature_vector(
                engagement_score * 50000, 
                0.1, 
                engagement_score
            )
            try:
                churn_proba = models['churn'].predict_proba([features])[0]
                churn_prob = churn_proba[1]
                
                if churn_prob > 0.6:
                    st.error(f"🔴 CRITICAL - User likely to churn: {churn_prob:.1%}")
                    st.warning("**Recommended Actions:**")
                    st.write("1. Send personalized retention offer")
                    st.write("2. Highlight new features and benefits")
                    st.write("3. Offer premium trial or discount")
                    st.write("4. Schedule 1-on-1 support call")
                elif churn_prob > 0.3:
                    st.warning(f"🟡 MEDIUM RISK - {churn_prob:.1%}")
                    st.write("**Recommended Actions:**")
                    st.write("1. Increase engagement notifications")
                    st.write("2. Share success stories and tips")
                    st.write("3. Offer loyalty rewards")
                else:
                    st.success(f"🟢 LOW RISK - User is engaged: {churn_prob:.1%}")
            except Exception as e:
                st.error(f"Error predicting churn: {e}")
    
    # ====================================================================
    # PAGE 9: COMPREHENSIVE ANALYSIS
    # ====================================================================
    elif page == "🔍 Comprehensive Analysis":
        st.markdown("## 🔍 Complete Financial Profile Analysis")
        
        st.write("Enter your complete financial information for a comprehensive analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_income = st.number_input("Monthly Income", 10000, 500000, 50000)
            monthly_spending = st.number_input("Monthly Spending", 5000, 400000, 30000)
            savings_goal = st.number_input("Monthly Savings Goal", 0, 100000, 10000)
        
        with col2:
            online_ratio = st.slider("Online Shopping %", 0, 100, 50)
            categories_used = st.slider("Categories Used", 1, 12, 6)
            days_since_transaction = st.slider("Days Since Last Transaction", 0, 30, 1)
        
        with col3:
            year_as_user = st.slider("Years as User", 0, 10, 1)
            transaction_frequency = st.number_input("Transactions per Month", 1, 1000, 100)
            goal_amount = st.number_input("Financial Goal (₹)", 10000, 10000000, 500000)
        
        if st.button("🚀 Generate Complete Analysis", key="complete"):
            features = create_feature_vector(
                monthly_spending, 
                abs(monthly_income - monthly_spending) / monthly_income if monthly_income > 0 else 0, 
                online_ratio / 100
            )
            
            try:
                # Get all predictions
                spending_pred = models['spending'].predict([features])[0]
                category_pred = models['category'].predict([features])[0]
                segment = models['segmentation'].predict([features])[0]
                risk_proba = models['risk'].predict_proba([features])[0]
                goal_proba = models['goal'].predict_proba([features])[0]
                churn_proba = models['churn'].predict_proba([features])[0]
                
                # Display dashboard
                st.markdown("### 📊 Analysis Dashboard")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Predicted Spending", 
                        f"₹{spending_pred:,.0f}", 
                        f"{(spending_pred/monthly_spending - 1)*100:+.1f}%" if monthly_spending > 0 else "N/A"
                    )
                
                with col2:
                    st.metric(
                        "Risk Level", 
                        ["🟢 Low", "🔴 High"][int(risk_proba[1] > 0.5)], 
                        f"{risk_proba[1]:.1%}"
                    )
                
                with col3:
                    # Check if predict_proba exists for segmentation
                    if hasattr(models['segmentation'], 'predict_proba'):
                        seg_confidence = max(models['segmentation'].predict_proba([features])[0]) * 100
                    else:
                        seg_confidence = 80
                    st.metric(
                        "User Segment", 
                        f"Segment {segment}", 
                        f"{seg_confidence:.0f}% match"
                    )
                
                with col4:
                    st.metric(
                        "Churn Risk", 
                        ["🟢 Low", "🟡 Med", "🔴 High"][min(2, int(churn_proba[1] * 3))], 
                        f"{churn_proba[1]:.1%}"
                    )
                
                # Savings analysis
                st.markdown("### 💰 Savings & Goal Analysis")
                
                monthly_surplus = monthly_income - monthly_spending - savings_goal
                months_to_goal = (goal_amount / savings_goal) if savings_goal > 0 else float('inf')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Monthly Surplus", f"₹{monthly_surplus:,.0f}")
                
                with col2:
                    st.metric(
                        "Months to Goal", 
                        f"{min(months_to_goal, 120):.0f}" if months_to_goal != float('inf') else "∞"
                    )
                
                with col3:
                    goal_achievement = goal_proba[1]
                    st.metric(
                        "Goal Probability", 
                        f"{goal_achievement:.1%}", 
                        "✅ On track" if goal_achievement > 0.7 else "⚠️ At risk"
                    )
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = []
                
                if risk_proba[1] > 0.5:
                    recommendations.append("⚠️ Reduce spending volatility - establish fixed budget")
                
                if churn_proba[1] > 0.6:
                    recommendations.append("📱 Increase app engagement - explore new features")
                
                if goal_proba[1] < 0.5:
                    recommendations.append("💰 Increase monthly savings - reassess spending patterns")
                
                if online_ratio > 70:
                    recommendations.append("💳 Consider cashback options for online purchases")
                
                if transaction_frequency < 50:
                    recommendations.append("📊 Increase tracking - monitor all expenses")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✅ Your financial health is excellent!")
            
            except Exception as e:
                st.error(f"Error generating analysis: {e}")
                st.exception(e)

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def create_feature_vector(spending, volatility, online_ratio, n_features=66):
    """Create feature vector for predictions - FIXED VERSION"""
    features = np.zeros(n_features)
    
    # Fill relevant features with deterministic values
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
    features[45] = 6.0  # ✅ FIXED: Use constant instead of random
    features[50] = 0.5
    features[55] = online_ratio * 0.6
    features[60] = 1 - volatility
    
    # ✅ FIXED: Fill remaining with deterministic values based on existing features
    for i in range(n_features):
        if features[i] == 0:
            # Use a deterministic value based on position and existing features
            features[i] = (spending / 1000 + volatility * 50 + online_ratio * 50 + i) % 100
    
    return features

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()