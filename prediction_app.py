"""
Streamlit app for financial predictions using trained models.
Users can input their transaction details and get real-time predictions.
Run: streamlit run prediction_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Financial Predictions", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 Financial Prediction Engine")
st.markdown("Enter your transaction details to get AI-powered financial predictions")

# ---- Model Loading ----
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_paths = {
        "spending": "artifacts/model_1_spending.pkl",
        "category": "artifacts/model_2_category.pkl",
        "anomaly": "artifacts/model_3_anomaly.pkl",
        "segmentation": "artifacts/model_4_segmentation.pkl",
        "risk": "artifacts/model_5_risk.pkl",
        "goal": "artifacts/model_6_goal.pkl",
        "churn": "artifacts/model_7_churn.pkl",
    }
    
    for model_name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[model_name] = joblib.load(path)
            else:
                st.warning(f"⚠️ Model {model_name} not found at {path}")
        except Exception as e:
            st.error(f"❌ Failed to load {model_name} model: {str(e)}")
    
    return models

# Load models
models = load_models()

if not models:
    st.error("❌ No trained models found! Please train models first using the training app.")
    st.info("1. Run `streamlit run app.py` to train models")
    st.info("2. Wait for training to complete")
    st.info("3. Come back here to make predictions")
    st.stop()

# ---- Sidebar: Input Section ----
st.sidebar.header("📝 Input Your Details")

with st.sidebar:
    # Transaction Information
    st.subheader("Transaction Information")
    
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    
    # Date selection
    transaction_date = st.date_input(
        "Transaction Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Transaction amount
    transaction_amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )
    
    # Category selection
    category = st.selectbox(
        "Spending Category",
        [
            "Food & Dining",
            "Entertainment",
            "Shopping",
            "Transportation",
            "Utilities",
            "Healthcare",
            "Education",
            "Other"
        ]
    )
    
    st.subheader("Historical Data (Monthly)")
    
    # Average monthly spending
    avg_monthly_spending = st.number_input(
        "Average Monthly Spending ($)",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )
    
    # Spending volatility
    spending_volatility = st.slider(
        "Spending Volatility (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher = more inconsistent spending"
    )
    
    # Account age (months)
    account_age_months = st.number_input(
        "Account Age (months)",
        min_value=1,
        value=12,
        step=1
    )
    
    # Number of transactions last month
    transactions_last_month = st.number_input(
        "Transactions Last Month",
        min_value=0,
        value=15,
        step=1
    )
    
    # Savings rate
    savings_rate = st.slider(
        "Savings Rate (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="% of income saved monthly"
    )
    
    st.subheader("Financial Goals")
    
    # Has savings goal
    has_savings_goal = st.checkbox("Has savings goal", value=True)
    
    # Goal achievement status
    goal_achievement_rate = st.slider(
        "Goal Achievement Rate (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=5,
        help="% of goals achieved previously"
    )
    
    predict_button = st.button("🔮 Get Predictions", type="primary", use_container_width=True)

# ---- Feature Engineering from User Input ----
def prepare_user_features(user_data):
    """Convert user input to model features"""
    features = pd.DataFrame({
        'user_id': [user_data['user_id']],
        'amount': [user_data['transaction_amount']],
        'amount_sum': [user_data['avg_monthly_spending']],
        'amount_mean': [user_data['avg_monthly_spending'] / max(user_data['transactions_last_month'], 1)],
        'amount_count': [user_data['transactions_last_month']],
        'spending_volatility': [user_data['spending_volatility']],
        'account_age_months': [user_data['account_age_months']],
        'savings_rate': [user_data['savings_rate']],
        'goal_achievement_rate': [user_data['goal_achievement_rate']],
    })
    
    # Drop non-numeric or ID columns for predictions
    X = features.drop(['user_id'], axis=1)
    return X

# ---- Predictions ----
if predict_button:
    user_data = {
        'user_id': int(user_id),
        'transaction_date': transaction_date,
        'transaction_amount': transaction_amount,
        'category': category,
        'avg_monthly_spending': avg_monthly_spending,
        'spending_volatility': spending_volatility,
        'account_age_months': account_age_months,
        'transactions_last_month': transactions_last_month,
        'savings_rate': savings_rate,
        'has_savings_goal': has_savings_goal,
        'goal_achievement_rate': goal_achievement_rate,
    }
    
    # Prepare features
    X_user = prepare_user_features(user_data)
    
    st.subheader("🎯 Prediction Results")
    
    # Create tabs for different predictions
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💵 Spending",
        "📊 Category",
        "⚠️ Anomaly",
        "👥 Segment",
        "🚨 Risk",
        "🎯 Goal",
        "👋 Churn"
    ])
    
    # ========== TAB 1: SPENDING PREDICTION ==========
    with tab1:
        st.write("**Predict next month's spending amount**")
        try:
            if "spending" in models:
                spending_pred = models["spending"].predict(X_user)[0]
                spending_pred_interval = models["spending"].predict_proba(X_user) if hasattr(models["spending"], 'predict_proba') else None
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Spending",
                        value=f"${spending_pred:,.2f}",
                        delta=f"${spending_pred - user_data['avg_monthly_spending']:+,.2f} vs avg"
                    )
                
                with col2:
                    st.metric(
                        label="Current Input Amount",
                        value=f"${transaction_amount:,.2f}",
                    )
                
                with col3:
                    pct_change = ((spending_pred - user_data['avg_monthly_spending']) / user_data['avg_monthly_spending'] * 100) if user_data['avg_monthly_spending'] > 0 else 0
                    st.metric(
                        label="Change from Average",
                        value=f"{pct_change:+.1f}%",
                    )
                
                if spending_pred > user_data['avg_monthly_spending'] * 1.2:
                    st.warning("⚠️ Predicted spending is significantly higher than your average. Consider reviewing this transaction.")
                elif spending_pred < user_data['avg_monthly_spending'] * 0.8:
                    st.success("✅ Predicted spending is lower than your average. Keep up the good spending habits!")
            else:
                st.error("Spending model not available")
        except Exception as e:
            st.error(f"Error in spending prediction: {str(e)}")
    
    # ========== TAB 2: CATEGORY FORECAST ==========
    with tab2:
        st.write("**Predicted spending by category next month**")
        try:
            if "category" in models:
                category_pred = models["category"].predict(X_user)[0]
                
                # Display category breakdown
                categories_list = [
                    "Food & Dining", "Entertainment", "Shopping",
                    "Transportation", "Utilities", "Healthcare", "Education", "Other"
                ]
                
                if isinstance(category_pred, np.ndarray) and len(category_pred) == len(categories_list):
                    category_df = pd.DataFrame({
                        'Category': categories_list,
                        'Predicted Amount': category_pred
                    }).sort_values('Predicted Amount', ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Top Categories:**")
                        st.dataframe(category_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.bar_chart(category_df.set_index('Category')['Predicted Amount'])
                else:
                    st.info(f"Category forecast: {category}")
            else:
                st.error("Category model not available")
        except Exception as e:
            st.error(f"Error in category prediction: {str(e)}")
    
    # ========== TAB 3: ANOMALY DETECTION ==========
    with tab3:
        st.write("**Is this transaction unusual?**")
        try:
            if "anomaly" in models:
                anomaly_pred = models["anomaly"].predict(X_user)[0]
                
                if anomaly_pred == -1:
                    st.error("🚨 **ANOMALY DETECTED!** This transaction is unusual compared to your pattern.")
                    st.write("Reasons this might be flagged:")
                    st.write("- Amount significantly different from typical transactions")
                    st.write("- Different time/category than usual")
                    st.write("- Unusual frequency of transactions")
                else:
                    st.success("✅ **NORMAL TRANSACTION** - This transaction matches your typical pattern.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Anomaly Score", f"{abs(anomaly_pred)}", help="Higher = more unusual")
                with col2:
                    st.metric("Status", "⚠️ Anomaly" if anomaly_pred == -1 else "✅ Normal")
            else:
                st.error("Anomaly model not available")
        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")
    
    # ========== TAB 4: USER SEGMENTATION ==========
    with tab4:
        st.write("**What user segment do you belong to?**")
        try:
            if "segmentation" in models:
                segment_pred = models["segmentation"].predict(X_user)[0]
                
                segment_names = {
                    0: "Conservative Saver",
                    1: "Moderate Spender",
                    2: "High Earner",
                    3: "Budget Conscious",
                    4: "Active Investor"
                }
                
                segment_name = segment_names.get(segment_pred, f"Segment {segment_pred}")
                
                st.info(f"🏷️ **Your Segment: {segment_name}**")
                
                segment_descriptions = {
                    0: "You maintain low spending and prioritize savings. Focus: Financial security",
                    1: "Balanced spending and saving habits. Focus: Financial stability",
                    2: "High income with moderate spending. Focus: Wealth growth",
                    3: "Careful with expenses and maximize savings. Focus: Frugality",
                    4: "Active in financial planning and investments. Focus: Wealth accumulation"
                }
                
                description = segment_descriptions.get(segment_pred, "")
                st.write(f"**Profile:** {description}")
            else:
                st.error("Segmentation model not available")
        except Exception as e:
            st.error(f"Error in user segmentation: {str(e)}")
    
    # ========== TAB 5: RISK ASSESSMENT ==========
    with tab5:
        st.write("**What's your financial risk level?**")
        try:
            if "risk" in models:
                risk_pred = models["risk"].predict(X_user)[0]
                risk_proba = models["risk"].predict_proba(X_user)[0] if hasattr(models["risk"], 'predict_proba') else None
                
                risk_level = "🚨 HIGH RISK" if risk_pred == 1 else "🟢 LOW RISK"
                
                st.metric("Risk Assessment", risk_level)
                
                if risk_pred == 1:
                    st.warning("⚠️ High financial risk detected!")
                    st.write("**Recommendations:**")
                    st.write("- Review spending patterns")
                    st.write("- Create an emergency fund")
                    st.write("- Consider consulting a financial advisor")
                else:
                    st.success("✅ Your financial risk is manageable")
                    st.write("**Continue to:**")
                    st.write("- Maintain current spending habits")
                    st.write("- Build emergency savings")
                    st.write("- Review goals regularly")
                
                if risk_proba is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Low Risk Probability", f"{risk_proba[0]*100:.1f}%")
                    with col2:
                        st.metric("High Risk Probability", f"{risk_proba[1]*100:.1f}%")
            else:
                st.error("Risk model not available")
        except Exception as e:
            st.error(f"Error in risk assessment: {str(e)}")
    
    # ========== TAB 6: GOAL ACHIEVEMENT ==========
    with tab6:
        st.write("**Will you achieve your financial goals?**")
        try:
            if "goal" in models:
                goal_pred = models["goal"].predict(X_user)[0]
                goal_proba = models["goal"].predict_proba(X_user)[0] if hasattr(models["goal"], 'predict_proba') else None
                
                achievement_status = "✅ LIKELY" if goal_pred == 1 else "⚠️ AT RISK"
                
                st.metric("Goal Achievement Forecast", achievement_status)
                
                if goal_pred == 1:
                    st.success("✅ You're on track to achieve your financial goals!")
                    st.write("**Your strengths:**")
                    st.write(f"- Consistent savings rate: {savings_rate}%")
                    st.write(f"- Past achievement rate: {goal_achievement_rate}%")
                    st.write(f"- Account stability: {(1 - spending_volatility)*100:.0f}% stable")
                else:
                    st.warning("⚠️ Your current path may not achieve goals")
                    st.write("**Action items:**")
                    st.write("- Increase savings rate")
                    st.write("- Reduce spending volatility")
                    st.write("- Set realistic milestones")
                
                if goal_proba is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Won't Achieve", f"{goal_proba[0]*100:.1f}%")
                    with col2:
                        st.metric("Will Achieve", f"{goal_proba[1]*100:.1f}%")
            else:
                st.error("Goal achievement model not available")
        except Exception as e:
            st.error(f"Error in goal achievement prediction: {str(e)}")
    
    # ========== TAB 7: CHURN PREDICTION ==========
    with tab7:
        st.write("**Likelihood of account inactivity**")
        try:
            if "churn" in models:
                churn_pred = models["churn"].predict(X_user)[0]
                churn_proba = models["churn"].predict_proba(X_user)[0] if hasattr(models["churn"], 'predict_proba') else None
                
                churn_status = "⚠️ LIKELY TO BE INACTIVE" if churn_pred == 1 else "✅ ACTIVE USER"
                
                st.metric("Churn Risk", churn_status)
                
                if churn_pred == 1:
                    st.warning("⚠️ Your account may become inactive")
                    st.write("**We recommend:**")
                    st.write("- Set up automated savings")
                    st.write("- Enable alerts for transactions")
                    st.write("- Review your financial goals")
                else:
                    st.success("✅ You appear to be an engaged user!")
                    st.write("**Keep it up with:**")
                    st.write("- Regular account reviews")
                    st.write("- Active financial management")
                    st.write("- Goal monitoring")
                
                if churn_proba is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Retention Probability", f"{churn_proba[0]*100:.1f}%")
                    with col2:
                        st.metric("Churn Probability", f"{churn_proba[1]*100:.1f}%")
            else:
                st.error("Churn model not available")
        except Exception as e:
            st.error(f"Error in churn prediction: {str(e)}")
    
    # ========== SUMMARY ========== 
    st.subheader("📋 Prediction Summary")
    
    summary_data = {
        'Metric': [
            'User ID',
            'Transaction Date',
            'Transaction Amount',
            'Category',
            'Avg Monthly Spending',
            'Spending Volatility',
            'Account Age',
            'Transactions Last Month',
            'Savings Rate',
            'Goal Achievement Rate'
        ],
        'Value': [
            user_id,
            transaction_date.strftime("%Y-%m-%d"),
            f"${transaction_amount:,.2f}",
            category,
            f"${avg_monthly_spending:,.2f}",
            f"{spending_volatility:.1f}",
            f"{account_age_months} months",
            transactions_last_month,
            f"{savings_rate}%",
            f"{goal_achievement_rate}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---- Footer ----
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 **Tip:** Enter your transaction details in the left sidebar and click "Get Predictions"</p>
    <p>🔄 All predictions are based on your historical patterns and the trained financial models</p>
    </div>
    """, unsafe_allow_html=True)
