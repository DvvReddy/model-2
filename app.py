# app.py
"""
Streamlit app to train & deploy the 7 financial models.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import zipfile
import io
from sklearn.model_selection import train_test_split
import traceback

st.set_page_config(page_title="Train All Models", layout="wide")

# ---- Helper: imports of your model classes ----
# Make sure models/ package is on PYTHONPATH; if not, adjust sys.path
app_dir = os.path.abspath(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.append(app_dir)

try:
    from models.model_1_spending_prediction import SpendingPredictionModel
    from models.model_2_category_forecast import CategoryForecastModel
    from models.model_3_anomaly_detection import AnomalyDetectionModel
    from models.model_4_user_segmentation import UserSegmentationModel
    from models.model_5_risk_assessment import RiskAssessmentModel
    from models.model_6_goal_achievement import GoalAchievementModel
    from models.model_7_churn_prediction import ChurnPredictionModel
except Exception as e:
    st.error("Failed to import model classes from models/ package. Check models package and PYTHONPATH.")
    st.exception(e)
    st.stop()

# ---- UI Layout ----
st.title("Train All 7 Financial Models — Streamlit")
st.markdown(
    """
This app trains all seven models on a Parquet-engineered dataset.  
You can upload a Parquet file or let the app look for `data/engineered_features_transaction_level.parquet`.
"""
)

with st.sidebar:
    st.header("Options")
    uploaded_file = st.file_uploader("Upload engineered features (Parquet)", type=["parquet", "parq"])
    use_local = st.checkbox("Use local data/engineered_features_transaction_level.parquet (if exists)", value=True)
    artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    random_seed = st.number_input("Random seed", value=42, step=1)
    run_button = st.button("Start training")

# ---- Session state containers ----
if "logs" not in st.session_state:
    st.session_state.logs = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "trained_files" not in st.session_state:
    st.session_state.trained_files = []

def log(msg, level="info"):
    st.session_state.logs.append(f"[{level.upper()}] {msg}")
    # keep log visible
    st.experimental_rerun() if False else None  # noop to avoid lint issues

def load_parquet_from_upload(uploaded):
    try:
        # read into pandas
        uploaded.seek(0)
        df = pd.read_parquet(uploaded)
        return df
    except Exception as e:
        raise

def load_parquet_from_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)

# ---- Data load preview area ----
data_load_col, preview_col = st.columns([1, 2])
with data_load_col:
    st.subheader("Data source")
    if uploaded_file is not None:
        st.write("Using uploaded file")
    else:
        st.write("Using local file if available")

    selected_path = os.path.join("data", "engineered_features_transaction_level.parquet")
    st.write("Local path:", selected_path)

with preview_col:
    st.subheader("Preview")
    df_preview = None
    try:
        if uploaded_file is not None:
            df_preview = load_parquet_from_upload(uploaded_file)
        elif use_local and os.path.exists(selected_path):
            df_preview = load_parquet_from_path(selected_path)
        else:
            st.info("No file selected and local file not present.")
        if df_preview is not None:
            st.write(f"Loaded DataFrame — {len(df_preview):,} rows × {len(df_preview.columns)} columns")
            st.dataframe(df_preview.head(50))
    except Exception as e:
        st.error("Failed to load Parquet.")
        st.exception(e)
        st.stop()

# ---- Core training pipeline ----
def prepare_features(df):
    # replicate your script's preparation steps (robust)
    df = df.copy()
    # date -> month
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M")
    elif "month" not in df.columns:
        df["month"] = 1

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {}
    for col in numeric_cols:
        if col not in ["user_id", "amount"]:
            agg_dict[col] = "mean"
    if "amount" in numeric_cols:
        agg_dict["amount"] = ["sum", "mean", "std", "count"]

    df_monthly = df.groupby(["user_id", "month"]).agg(agg_dict).reset_index()
    # flatten
    df_monthly.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_monthly.columns.values
    ]

    # X_monthly
    X_monthly = df_monthly.select_dtypes(include=[np.number]).copy()

    # target spending
    if "amount_sum" in X_monthly.columns:
        y_monthly_spending = X_monthly["amount_sum"].copy()
        X_monthly = X_monthly.drop("amount_sum", axis=1)
    elif "amount_mean" in X_monthly.columns:
        y_monthly_spending = X_monthly["amount_mean"].copy() * X_monthly.get("amount_count", 1)
        X_monthly = X_monthly.drop("amount_mean", axis=1, errors="ignore")
        X_monthly = X_monthly.drop("amount_count", axis=1, errors="ignore")
    else:
        # fallback
        if X_monthly.shape[1] < 2:
            raise ValueError("Not enough numeric columns to create a target and features.")
        y_monthly_spending = X_monthly.iloc[:, 0].copy()
        X_monthly = X_monthly.iloc[:, 1:]

    valid_idx = ~y_monthly_spending.isna()
    y_monthly_spending = y_monthly_spending[valid_idx]
    X_monthly = X_monthly.loc[valid_idx, :]

    y_monthly_risk = (X_monthly.get("spending_volatility", pd.Series(0, index=X_monthly.index)) > 0.5).astype(int)
    y_monthly_goal = (y_monthly_spending < y_monthly_spending.rolling(3, min_periods=1).mean()).astype(int).fillna(0)

    X_user = X_monthly.groupby(df_monthly.loc[valid_idx, "user_id"]).mean()

    X_transaction = df.select_dtypes(include=[np.number]).copy()
    for col in ["user_id", "transaction_id", "amount", "month"]:
        X_transaction = X_transaction.drop(col, axis=1, errors="ignore")

    return {
        "df_monthly": df_monthly,
        "X_monthly": X_monthly,
        "y_spending": y_monthly_spending,
        "y_risk": y_monthly_risk,
        "y_goal": y_monthly_goal,
        "X_user": X_user,
        "X_transaction": X_transaction,
    }

def save_model(model_obj, path):
    try:
        model_obj.save(path)
    except Exception:
        # Try pickle as fallback
        import joblib
        joblib.dump(model_obj, path)

def run_training_pipeline(df, artifacts_dir, seed=42, progress_callback=None, log_callback=None):
    """Runs training for all 7 models. Returns metrics dict and saved model file list."""
    np.random.seed(seed)
    results = {}
    saved_files = []

    # Prepare
    if log_callback: log_callback("Preparing features...")
    feats = prepare_features(df)

    X_monthly = feats["X_monthly"]
    y_spending = feats["y_spending"]
    y_risk = feats["y_risk"]
    y_goal = feats["y_goal"]
    X_user = feats["X_user"]
    X_transaction = feats["X_transaction"]

    total_steps = 7
    step = 0

    # 1 Spending prediction
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Spending Prediction ({step}/{total_steps})")
    try:
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
            X_monthly, y_spending, test_size=0.2, random_state=seed
        )
        model1 = SpendingPredictionModel(rf_estimators=100, gb_estimators=100)
        model1.train(X_train_1, y_train_1)
        metrics1 = model1.evaluate(X_test_1, y_test_1)
        p = os.path.join(artifacts_dir, "model_1_spending.pkl")
        save_model(model1, p)
        saved_files.append(p)
        results["model_1"] = metrics1
        if log_callback: log_callback(f"Model 1 done — MAE: {metrics1.get('mae')}, R2: {metrics1.get('r2')}")
    except Exception as e:
        if log_callback: log_callback(f"Model 1 failed: {e}", level="error")
        results["model_1"] = {"error": str(e), "trace": traceback.format_exc()}

    # 2 Category forecast
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Category Forecast ({step}/{total_steps})")
    try:
        category_cols = [col for col in df.columns if col.startswith("cat_")]
        if len(category_cols) > 1:
            y_categories = df[category_cols + ["user_id", "month"]].groupby(["user_id", "month"])[category_cols].sum().reset_index()
            y_categories = y_categories.drop(["user_id", "month"], axis=1)
            # if shapes mismatch, try to align by index through join on user_id+month
            if len(y_categories) == len(X_monthly):
                X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
                    X_monthly, y_categories, test_size=0.2, random_state=seed
                )
                model2 = CategoryForecastModel(n_estimators=100)
                model2.train(X_train_2, y_train_2)
                metrics2 = model2.evaluate(X_test_2, y_test_2)
                p = os.path.join(artifacts_dir, "model_2_category.pkl")
                save_model(model2, p)
                saved_files.append(p)
                results["model_2"] = metrics2
                if log_callback: log_callback(f"Model 2 done — avg R2: {metrics2.get('avg_r2')}")
            else:
                results["model_2"] = {"skipped": "dimension_mismatch"}
                if log_callback: log_callback("Model 2 skipped due to dimension mismatch", level="warning")
        else:
            results["model_2"] = {"skipped": "not_enough_categories"}
            if log_callback: log_callback("Model 2 skipped: not enough category columns", level="warning")
    except Exception as e:
        if log_callback: log_callback(f"Model 2 failed: {e}", level="error")
        results["model_2"] = {"error": str(e), "trace": traceback.format_exc()}

    # 3 Anomaly detection
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Anomaly Detection ({step}/{total_steps})")
    try:
        model3 = AnomalyDetectionModel(contamination=0.05)
        model3.train(X_transaction)
        p = os.path.join(artifacts_dir, "model_3_anomaly.pkl")
        save_model(model3, p)
        saved_files.append(p)
        results["model_3"] = {"transactions_analyzed": len(X_transaction)}
        if log_callback: log_callback(f"Model 3 done — transactions: {len(X_transaction):,}")
    except Exception as e:
        if log_callback: log_callback(f"Model 3 failed: {e}", level="error")
        results["model_3"] = {"error": str(e), "trace": traceback.format_exc()}

    # 4 User segmentation
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training User Segmentation ({step}/{total_steps})")
    try:
        model4 = UserSegmentationModel(n_clusters=5, method="kmeans")
        model4.train(X_user)
        metrics4 = model4.evaluate(X_user)
        p = os.path.join(artifacts_dir, "model_4_segmentation.pkl")
        save_model(model4, p)
        saved_files.append(p)
        results["model_4"] = metrics4
        if log_callback: log_callback(f"Model 4 done — Silhouette: {metrics4.get('silhouette_score')}")
    except Exception as e:
        if log_callback: log_callback(f"Model 4 failed: {e}", level="error")
        results["model_4"] = {"error": str(e), "trace": traceback.format_exc()}

    # 5 Risk assessment
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Risk Assessment ({step}/{total_steps})")
    try:
        X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(
            X_monthly, y_risk, test_size=0.2, random_state=seed
        )
        model5 = RiskAssessmentModel(max_depth=6, learning_rate=0.1)
        model5.train(X_train_5, y_train_5)
        metrics5 = model5.evaluate(X_test_5, y_test_5)
        p = os.path.join(artifacts_dir, "model_5_risk.pkl")
        save_model(model5, p)
        saved_files.append(p)
        results["model_5"] = metrics5
        if log_callback: log_callback(f"Model 5 done — Accuracy: {metrics5.get('accuracy')}")
    except Exception as e:
        if log_callback: log_callback(f"Model 5 failed: {e}", level="error")
        results["model_5"] = {"error": str(e), "trace": traceback.format_exc()}

    # 6 Goal achievement
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Goal Achievement ({step}/{total_steps})")
    try:
        X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(
            X_monthly, y_goal, test_size=0.2, random_state=seed
        )
        model6 = GoalAchievementModel(n_estimators=100, max_depth=7)
        model6.train(X_train_6, y_train_6)
        metrics6 = model6.evaluate(X_test_6, y_test_6)
        p = os.path.join(artifacts_dir, "model_6_goal.pkl")
        save_model(model6, p)
        saved_files.append(p)
        results["model_6"] = metrics6
        if log_callback: log_callback(f"Model 6 done — Accuracy: {metrics6.get('accuracy')}")
    except Exception as e:
        if log_callback: log_callback(f"Model 6 failed: {e}", level="error")
        results["model_6"] = {"error": str(e), "trace": traceback.format_exc()}

    # 7 Churn prediction (note: original used y_monthly_goal as a placeholder)
    step += 1
    if progress_callback: progress_callback(step, total_steps, f"Training Churn Prediction ({step}/{total_steps})")
    try:
        X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(
            X_monthly, y_goal, test_size=0.2, random_state=seed
        )
        model7 = ChurnPredictionModel(n_estimators=100)
        model7.train(X_train_7, y_train_7)
        metrics7 = model7.evaluate(X_test_7, y_test_7)
        p = os.path.join(artifacts_dir, "model_7_churn.pkl")
        save_model(model7, p)
        saved_files.append(p)
        results["model_7"] = metrics7
        if log_callback: log_callback(f"Model 7 done — Accuracy: {metrics7.get('accuracy')}")
    except Exception as e:
        if log_callback: log_callback(f"Model 7 failed: {e}", level="error")
        results["model_7"] = {"error": str(e), "trace": traceback.format_exc()}

    return results, saved_files

# ---- Training invocation ----
if run_button:
    if df_preview is None:
        st.error("No data loaded. Upload a Parquet or enable/use the local file.")
    else:
        # Clear logs
        st.session_state.logs = []
        st.session_state.metrics = {}
        st.session_state.trained_files = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_box = st.empty()

        def progress_cb(step, total, message=None):
            frac = int((step / total) * 100)
            progress_bar.progress(frac)
            if message:
                status_text.info(message)

        def log_cb(message, level="info"):
            st.session_state.logs.append(f"[{level.upper()}] {message}")
            # update log box
            with log_box.container():
                st.write("\n".join(st.session_state.logs[-200:]))

        try:
            with st.spinner("Training... this runs sequentially and may take time depending on data and models"):
                metrics, saved_files = run_training_pipeline(
                    df_preview, artifacts_dir=artifacts_dir, seed=int(random_seed),
                    progress_callback=progress_cb, log_callback=log_cb
                )

            st.success("Training run finished.")
            st.session_state.metrics = metrics
            st.session_state.trained_files = saved_files

            # Show metrics
            st.subheader("Metrics / Results")
            for k, v in metrics.items():
                st.write(f"**{k}**")
                st.json(v)

            # Show saved model files
            st.subheader("Saved artifact files")
            for f in saved_files:
                st.write(f"• {f} — {os.path.getsize(f):,} bytes")

            # Provide ZIP download
            if saved_files:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in saved_files:
                        arcname = os.path.basename(fp)
                        zf.write(fp, arcname)
                buf.seek(0)
                st.download_button("Download all models (zip)", data=buf, file_name="models_artifacts.zip", mime="application/zip")
        except Exception as e:
            st.error("Training failed with an exception.")
            st.exception(e)
            st.write("Traceback (latest logs):")
            st.write("\n".join(st.session_state.logs[-50:]))

# ---- Logs viewer ----
st.subheader("Run logs")
st.text_area("Logs", value="\n".join(st.session_state.logs[-500:]), height=300)
