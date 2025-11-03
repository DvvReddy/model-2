# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FinSight AI", page_icon="💰", layout="wide")

st.title("🚀 FinSight AI – Upload & Predict All 7 Models")
st.markdown("**Upload your Parquet → Get Spending, Risk, Churn & More in 10 Seconds!**")

# --------------------------- SAFE MODEL LOADER ---------------------------
@st.cache_resource
def load_models_safely():
    """Load models with error handling for version mismatches"""
    paths = {
        "spending": "artifacts/model_1_spending.pkl",
        "category": "artifacts/model_2_category.pkl", 
        "anomaly": "artifacts/model_3_anomaly.pkl",
        "segmentation": "artifacts/model_4_segmentation.pkl",
        "risk": "artifacts/model_5_risk.pkl",
        "goal": "artifacts/model_6_goal.pkl",
        "churn": "artifacts/model_7_churn.pkl",
    }
    
    models = {}
    for name, path in paths.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
                st.success(f"✅ {name.replace('_', ' ').title()}")
            else:
                st.warning(f"⚠️ Missing: {path}")
        except Exception as e:
            st.error(f"❌ {name}: {str(e)[:100]}")
    
    return models

# Load with safety net
models = load_models_safely()
available_models = {k: v for k, v in models.items() if k in models}

st.divider()

# --------------------------- FILE UPLOADER ---------------------------
uploaded_file = st.file_uploader(
    "📁 Upload `engineered_features_transaction_level.parquet`",
    type=["parquet"],
    help="Must match training format"
)

if uploaded_file and len(available_models) > 0:
    with st.spinner("🔄 Processing your data..."):
        try:
            # Load & preprocess EXACTLY like training
            df = pd.read_parquet(uploaded_file)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.to_period('M')
            
            # Monthly aggregation (exact match to training)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            agg_dict = {}
            for col in numeric_cols:
                if col not in ['user_id', 'amount']:
                    agg_dict[col] = 'mean'
            if 'amount' in df.columns:
                agg_dict['amount'] = ['sum', 'mean', 'std', 'count']
            
            monthly = df.groupby(['user_id', 'month']).agg(agg_dict).reset_index()
            monthly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                             for col in monthly.columns]
            
            # Prepare features
            X_monthly = monthly.select_dtypes(include=[np.number])
            
            # Remove target leaks
            target_cols = ['amount_sum', 'amount_mean', 'amount_std', 'amount_count']
            X_monthly = X_monthly.drop(columns=[c for c in target_cols if c in X_monthly.columns], 
                                     errors='ignore')
            
            # Handle missing values
            X_monthly = X_monthly.fillna(X_monthly.mean())
            
            st.success(f"✅ Processed {len(X_monthly):,} monthly records from {df['user_id'].nunique():,} users")
            
        except Exception as e:
            st.error(f"❌ Data processing failed: {e}")
            st.stop()

    # --------------------------- PREDICTIONS DASHBOARD ---------------------------
    tab1, tab2, tab3 = st.tabs(["💸 Spending & Categories", "⚠️ Risk & Goal", "🚪 Churn & Segment"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if "spending" in available_models:
                pred_spend = available_models["spending"].predict(X_monthly)
                st.metric("Avg Next Month Spend", f"₹{pred_spend.mean():,.0f}", 
                         f"±₹{pred_spend.std():,.0f}")
                fig = px.histogram(pred_spend, nbins=30, title="Spending Predictions")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "category" in available_models:
                try:
                    cat_pred = available_models["category"].predict(X_monthly)
                    if hasattr(available_models["category"], 'category_names'):
                        cat_names = available_models["category"].category_names
                    else:
                        cat_names = [f"cat_{i}" for i in range(cat_pred.shape[1])]
                    
                    avg_cat = pd.DataFrame(cat_pred, columns=cat_names).mean()
                    fig = px.bar(x=avg_cat.index, y=avg_cat.values, 
                               title="Avg Category Spending Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Category model needs adjustment")

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if "risk" in available_models:
                try:
                    risk_scores = available_models["risk"].predict_proba(X_monthly)[:, 1]
                    high_risk = (risk_scores > 0.5).sum()
                    st.metric("High Risk Users", f"{high_risk}/{len(risk_scores)}", 
                             f"{high_risk/len(risk_scores):.1%}")
                    fig = px.histogram(risk_scores, nbins=20, title="Risk Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Risk scores unavailable")
        
        with col2:
            if "goal" in available_models:
                try:
                    goal_scores = available_models["goal"].predict_proba(X_monthly)[:, 1]
                    achievers = (goal_scores > 0.5).sum()
                    st.metric("Likely Goal Achievers", f"{achievers}/{len(goal_scores)}")
                    fig = px.histogram(goal_scores, nbins=20, title="Goal Probability")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Goal predictions unavailable")

    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if "churn" in available_models:
                try:
                    churn_scores = available_models["churn"].predict_proba(X_monthly)[:, 1]
                    st.metric("Avg Churn Risk", f"{churn_scores.mean():.1%}")
                    fig = px.histogram(churn_scores, nbins=20, title="Churn Risk Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Churn predictions unavailable")
        
        with col2:
            if "segmentation" in available_models:
                try:
                    # User-level segments
                    X_user = X_monthly.groupby(monthly['user_id']).mean()
                    segments = available_models["segmentation"].predict(X_user)
                    seg_df = pd.DataFrame({'segment': segments}).value_counts().reset_index(name='count')
                    fig = px.pie(seg_df, values='count', names='segment', title="User Segments")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Segmentation unavailable")
            
            if "anomaly" in available_models:
                st.info("🕵️ Anomaly detection ready for transaction-level data")

    st.balloons()
    st.success("🎉 All predictions complete!")

elif len(available_models) == 0:
    st.error("❌ No models found in /artifacts folder!")
    st.info("👈 Upload your 7 .pkl files from `train_all_models.py`")

else:
    st.info("👈 Upload your Parquet file to get predictions!")
    st.markdown("""
    ### 🚀 Quick Start
    1. **Upload** `engineered_features_transaction_level.parquet`
    2. **View** predictions for all your users instantly
    3. **Share** this link with your team!
    """)

st.markdown("---")
st.caption("**FinSight AI** – Powered by 7 ML Models | Deployed on Streamlit Cloud")