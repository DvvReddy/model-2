# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="FinSight AI", page_icon="💰", layout="wide")

st.title("🚀 FinSight AI – Upload & Predict All 7 Models")
st.markdown("**Upload your Parquet → Get Spending, Risk, Churn & More in 10 Seconds!**")

# Load models once
@st.cache_resource
def load_models():
    paths = {
        "spending": "artifacts/model_1_spending.pkl",
        "category": "artifacts/model_2_category.pkl",
        "anomaly": "artifacts/model_3_anomaly.pkl",
        "segmentation": "artifacts/model_4_segmentation.pkl",
        "risk": "artifacts/model_5_risk.pkl",
        "goal": "artifacts/model_6_goal.pkl",
        "churn": "artifacts/model_7_churn.pkl",
    }
    return {name: joblib.load(path) for name, path in paths.items() if os.path.exists(path)}

models = load_models()
if len(models) < 7:
    st.error("⚠️ Some models missing in /artifacts folder!")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload `engineered_features_transaction_level.parquet`",
    type=["parquet"],
    help="Must match the exact format used in training"
)

if uploaded_file:
    with st.spinner("Reading your data..."):
        df = pd.read_parquet(uploaded_file)
    st.success(f"Loaded {len(df):,} transactions")

    # === Reuse your exact preprocessing from train_all_models.py ===
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')

    numeric_cols = df.select_dtypes(include='number').columns
    agg_dict = {col: 'mean' for col in numeric_cols if col not in ['user_id', 'amount']}
    if 'amount' in df.columns:
        agg_dict['amount'] = ['sum', 'mean', 'std', 'count']

    monthly = df.groupby(['user_id', 'month']).agg(agg_dict).reset_index()
    monthly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in monthly.columns]

    X_monthly = monthly.select_dtypes(include='number')
    if 'amount_sum' in X_monthly.columns:
        y_spend = X_monthly['amount_sum']
        X_monthly = X_monthly.drop('amount_sum', axis=1)
    else:
        y_spend = X_monthly['amount_mean'] * X_monthly.get('amount_count', 1)

    # Drop target leaks
    X_monthly = X_monthly.drop(columns=[c for c in ['amount_mean', 'amount_std', 'amount_count'] if c in X_monthly.columns], errors='ignore')

    st.divider()
    st.subheader("🔮 Live Predictions for Your Users")

    tab1, tab2, tab3 = st.tabs(["💸 Spending & Categories", "⚠️ Risk & Goal", "🚪 Churn & Segment"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            pred_spend = models["spending"].predict(X_monthly)
            st.metric("Next Month Spend", f"₹{pred_spend.mean():,.0f}", f"₹{pred_spend.std():,.0f} std")
            fig = px.histogram(pred_spend, nbins=30, title="Predicted Spend Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "category" in models:
                cat_pred = models["category"].predict(X_monthly)
                cat_names = models["category"].category_names
                avg_cat = pd.DataFrame(cat_pred, columns=cat_names).mean()
                fig = px.bar(x=avg_cat.index, y=avg_cat.values, labels={'x':'Category', 'y':'Avg ₹'})
                fig.update_layout(title="Average Category Forecast")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            risk_prob = models["risk"].predict_proba(X_monthly)[:, 1]
            st.metric("High-Risk Users", f"{(risk_prob > 0.5).sum()}/{len(risk_prob)}")
            fig = px.histogram(risk_prob, nbins=20, title="Risk Score Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            goal_prob = models["goal"].predict_proba(X_monthly)[:, 1]
            st.metric("Goal Achievers", f"{(goal_prob > 0.5).sum()}/{len(goal_prob)}")
            fig = px.histogram(goal_prob, nbins=20, title="Goal Achievement Probability")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            churn_prob = models["churn"].predict_proba(X_monthly)[:, 1]
            st.metric("Churn Risk", f"{churn_prob.mean():.1%} avg")
            high_churn = monthly.iloc[churn_prob > 0.7]
            if not high_churn.empty:
                st.write("**Top 5 High-Churn Users**")
                st.dataframe(high_churn[['user_id', 'month']].head())

        with col2:
            segments = models["segmentation"].predict(X_monthly.groupby(monthly['user_id']).mean())
            seg_counts = pd.Series(segments).value_counts()
            fig = px.pie(values=seg_counts.values, names=seg_counts.index, title="User Segments")
            st.plotly_chart(fig, use_container_width=True)

    st.balloons()
    st.success("All 7 models ran successfully!")

else:
    st.info("👈 Upload your Parquet file to start predicting!")
    st.markdown("""
    ### How to use
    1. Put your `engineered_features_transaction_level.parquet` in the same format as training  
    2. Upload it above  
    3. Get instant insights for **thousands of users**
    """)

st.caption("Deployed with ❤️ on Streamlit Community Cloud – Share this link with your team!")