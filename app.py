"""
Streamlit app to train & deploy the 7 financial models.
Run: streamlit run app.py
FIXED VERSION: Resolves import errors, deprecated functions, and file handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
import zipfile
import io
import joblib
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Train All Models", layout="wide")

# ---- Pre-import Dependency Check ----
missing_deps = []
try:
    import lightgbm
except ImportError:
    missing_deps.append("lightgbm")

try:
    import xgboost
except ImportError:
    missing_deps.append("xgboost")

if missing_deps:
    st.error("❌ Missing ML dependencies for model training!")
    st.code(f"pip install {' '.join(missing_deps)}", language="bash")
    st.stop()

# ---- Helper: imports of your model classes ----
app_dir = os.path.abspath(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from models.model_1_spending_prediction import SpendingPredictionModel
    from models.model_2_category_forecast import CategoryForecastModel
    from models.model_3_anomaly_detection import AnomalyDetectionModel
    from models.model_4_user_segmentation import UserSegmentationModel
    from models.model_5_risk_assessment import RiskAssessmentModel
    from models.model_6_goal_achievement import GoalAchievementModel
    from models.model_7_churn_prediction import ChurnPredictionModel
except Exception as e:
    st.error("❌ Failed to import model classes from models/ package.")
    st.error(f"**Error:** {str(e)}")
    st.info("**Troubleshooting:**")
    st.write("1. Verify `models/` folder exists with all 7 model files")
    st.write("2. Check `models/__init__.py` exists")
    st.write("3. Run: `pip install -r requirements.txt`")
    st.write("4. Verify LightGBM and XGBoost are installed")
    st.error(f"**Full traceback:**")
    st.code(traceback.format_exc())
    st.stop()

# ---- UI Layout ----
st.title("🎯 Train All 7 Financial Models")
st.markdown(
    """
    This app trains all seven financial models on a Parquet-engineered dataset.  
    You can upload a Parquet file or use the local dataset.
    """
)

# ---- Sidebar Configuration ----
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader(
        "Upload engineered features (Parquet)", 
        type=["parquet", "parq"]
    )
    use_local = st.checkbox(
        "Use local data/engineered_features_transaction_level.parquet", 
        value=True
    )
    artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    random_seed = st.number_input("Random seed", value=42, step=1, min_value=0)
    test_size = st.slider(
        "Train/Test split ratio", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.2, 
        step=0.05
    )
    run_button = st.button("▶️ Start Training", type="primary")

# ---- Session State Initialization ----
if "logs" not in st.session_state:
    st.session_state.logs = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "trained_files" not in st.session_state:
    st.session_state.trained_files = []

# ---- Helper Functions ----
def log(msg, level="info"):
    """Add timestamped message to session state logs"""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level.upper()}] {msg}"
    st.session_state.logs.append(formatted_msg)

def load_parquet_from_upload(uploaded):
    """Load Parquet file from uploaded file object"""
    try:
        uploaded.seek(0)
        df = pd.read_parquet(io.BytesIO(uploaded.read()))
        return df
    except Exception as e:
        st.error(f"❌ Failed to read uploaded Parquet file: {str(e)}")
        raise

def load_parquet_from_path(path):
    """Load Parquet file from local file path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.error(f"❌ Failed to read Parquet from {path}: {str(e)}")
        raise

def save_model(model_obj, path):
    """Save model to disk using joblib"""
    try:
        if hasattr(model_obj, 'save') and callable(model_obj.save):
            model_obj.save(path)
        else:
            joblib.dump(model_obj, path)
        return True
    except Exception as e:
        log(f"Failed to save model to {path}: {str(e)}", level="error")
        raise

# ---- Data Loading & Preview Section ----
st.subheader("📊 Data Source & Preview")
data_col, preview_col = st.columns([1, 2])

with data_col:
    st.write("**Source Configuration**")
    if uploaded_file is not None:
        st.write("✅ Using uploaded file")
    elif use_local:
        st.write("✅ Configured to use local file")
    else:
        st.write("⚠️ No data source selected")
    
    selected_path = os.path.join("data", "engineered_features_transaction_level.parquet")
    st.write(f"Local path: `{selected_path}`")

with preview_col:
    st.write("**DataFrame Preview**")
    df_preview = None
    try:
        if uploaded_file is not None:
            df_preview = load_parquet_from_upload(uploaded_file)
        elif use_local and os.path.exists(selected_path):
            df_preview = load_parquet_from_path(selected_path)
        else:
            st.info("ℹ️ No file selected. Upload a Parquet file to proceed.")
        
        if df_preview is not None:
            st.success(f"✅ Loaded: {len(df_preview):,} rows × {len(df_preview.columns)} columns")
            st.dataframe(df_preview.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to load Parquet: {str(e)}")
        df_preview = None

# ---- Feature Preparation Function ----
def prepare_features(df):
    """Prepare and aggregate features for all models"""
    try:
        df = df.copy()
        
        # Handle date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date"].dt.to_period("M")
        elif "month" not in df.columns:
            df["month"] = 1
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {}
        
        for col in numeric_cols:
            if col not in ["user_id", "amount"]:
                agg_dict[col] = "mean"
        
        if "amount" in numeric_cols:
            agg_dict["amount"] = ["sum", "mean", "std", "count"]
        
        # Aggregate to monthly level
        df_monthly = df.groupby(["user_id", "month"]).agg(agg_dict).reset_index()
        
        # Flatten multi-level column names
        df_monthly.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col 
            for col in df_monthly.columns.values
        ]
        
        # Extract numeric features
        X_monthly = df_monthly.select_dtypes(include=[np.number]).copy()
        
        # Create spending target
        if "amount_sum" in X_monthly.columns:
            y_monthly_spending = X_monthly["amount_sum"].copy()
            X_monthly = X_monthly.drop("amount_sum", axis=1)
        elif "amount_mean" in X_monthly.columns:
            y_monthly_spending = X_monthly["amount_mean"].copy() * X_monthly.get("amount_count", 1)
            X_monthly = X_monthly.drop("amount_mean", axis=1, errors="ignore")
            X_monthly = X_monthly.drop("amount_count", axis=1, errors="ignore")
        else:
            if X_monthly.shape[1] < 2:
                raise ValueError("Not enough numeric columns for target and features")
            y_monthly_spending = X_monthly.iloc[:, 0].copy()
            X_monthly = X_monthly.iloc[:, 1:]
        
        # Remove NaN values
        valid_idx = ~y_monthly_spending.isna()
        y_monthly_spending = y_monthly_spending[valid_idx]
        X_monthly = X_monthly.loc[valid_idx, :]
        
        # Create additional targets
        y_monthly_risk = (
            X_monthly.get("spending_volatility", pd.Series(0, index=X_monthly.index)) > 0.5
        ).astype(int)
        
        y_monthly_goal = (
            y_monthly_spending < y_monthly_spending.rolling(3, min_periods=1).mean()
        ).astype(int).fillna(0)
        
        # User-level aggregation
        X_user = X_monthly.groupby(df_monthly.loc[valid_idx, "user_id"]).mean()
        
        # Transaction-level features
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
    except Exception as e:
        log(f"Feature preparation failed: {str(e)}", level="error")
        raise

# ---- Core Training Pipeline ----
def run_training_pipeline(df, artifacts_dir, seed=42, test_size=0.2, 
                         progress_callback=None, log_callback=None):
    """Train all 7 models sequentially"""
    np.random.seed(seed)
    results = {}
    saved_files = []
    
    try:
        # Prepare features
        if log_callback:
            log_callback("🔧 Preparing features from dataset...")
        
        feats = prepare_features(df)
        
        X_monthly = feats["X_monthly"]
        y_spending = feats["y_spending"]
        y_risk = feats["y_risk"]
        y_goal = feats["y_goal"]
        X_user = feats["X_user"]
        X_transaction = feats["X_transaction"]
        
        total_steps = 7
        
        # ========== MODEL 1: Spending Prediction ==========
        step = 1
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Spending Prediction")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Spending Prediction Model...")
            
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
                X_monthly, y_spending, test_size=test_size, random_state=seed
            )
            model1 = SpendingPredictionModel(rf_estimators=100, gb_estimators=100)
            model1.train(X_train_1, y_train_1)
            metrics1 = model1.evaluate(X_test_1, y_test_1)
            
            p1 = os.path.join(artifacts_dir, "model_1_spending.pkl")
            save_model(model1, p1)
            saved_files.append(p1)
            results["model_1"] = metrics1
            
            mae = metrics1.get('mae', 'N/A')
            r2 = metrics1.get('r2', 'N/A')
            if log_callback:
                log_callback(f"✅ Model 1 complete — MAE: {mae}, R²: {r2}")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 1 failed: {str(e)}", level="error")
            results["model_1"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 2: Category Forecast ==========
        step = 2
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Category Forecast")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Category Forecast Model...")
            
            category_cols = [col for col in df.columns if col.startswith("cat_")]
            if len(category_cols) > 1:
                y_categories = df[category_cols + ["user_id", "month"]].groupby(
                    ["user_id", "month"]
                )[category_cols].sum().reset_index()
                y_categories = y_categories.drop(["user_id", "month"], axis=1)
                
                if len(y_categories) == len(X_monthly):
                    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
                        X_monthly, y_categories, test_size=test_size, random_state=seed
                    )
                    model2 = CategoryForecastModel(n_estimators=100)
                    model2.train(X_train_2, y_train_2)
                    metrics2 = model2.evaluate(X_test_2, y_test_2)
                    
                    p2 = os.path.join(artifacts_dir, "model_2_category.pkl")
                    save_model(model2, p2)
                    saved_files.append(p2)
                    results["model_2"] = metrics2
                    
                    avg_r2 = metrics2.get('avg_r2', 'N/A')
                    if log_callback:
                        log_callback(f"✅ Model 2 complete — Avg R²: {avg_r2}")
                else:
                    results["model_2"] = {"skipped": "dimension_mismatch"}
                    if log_callback:
                        log_callback("⚠️ Model 2 skipped: dimension mismatch", level="warning")
            else:
                results["model_2"] = {"skipped": "not_enough_categories"}
                if log_callback:
                    log_callback("⚠️ Model 2 skipped: insufficient category columns", level="warning")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 2 failed: {str(e)}", level="error")
            results["model_2"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 3: Anomaly Detection ==========
        step = 3
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Anomaly Detection")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Anomaly Detection Model...")
            
            if len(X_transaction) > 0:
                model3 = AnomalyDetectionModel(contamination=0.05)
                model3.train(X_transaction)
                
                p3 = os.path.join(artifacts_dir, "model_3_anomaly.pkl")
                save_model(model3, p3)
                saved_files.append(p3)
                results["model_3"] = {"transactions_analyzed": len(X_transaction)}
                
                if log_callback:
                    log_callback(f"✅ Model 3 complete — Transactions analyzed: {len(X_transaction):,}")
            else:
                results["model_3"] = {"skipped": "no_transaction_data"}
                if log_callback:
                    log_callback("⚠️ Model 3 skipped: no transaction data", level="warning")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 3 failed: {str(e)}", level="error")
            results["model_3"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 4: User Segmentation ==========
        step = 4
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training User Segmentation")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training User Segmentation Model...")
            
            if len(X_user) > 5:
                model4 = UserSegmentationModel(n_clusters=5, method="kmeans")
                model4.train(X_user)
                metrics4 = model4.evaluate(X_user)
                
                p4 = os.path.join(artifacts_dir, "model_4_segmentation.pkl")
                save_model(model4, p4)
                saved_files.append(p4)
                results["model_4"] = metrics4
                
                silhouette = metrics4.get('silhouette_score', 'N/A')
                if log_callback:
                    log_callback(f"✅ Model 4 complete — Silhouette Score: {silhouette}")
            else:
                results["model_4"] = {"skipped": "insufficient_users"}
                if log_callback:
                    log_callback("⚠️ Model 4 skipped: insufficient user data", level="warning")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 4 failed: {str(e)}", level="error")
            results["model_4"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 5: Risk Assessment ==========
        step = 5
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Risk Assessment")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Risk Assessment Model...")
            
            X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(
                X_monthly, y_risk, test_size=test_size, random_state=seed
            )
            model5 = RiskAssessmentModel(max_depth=6, learning_rate=0.1)
            model5.train(X_train_5, y_train_5)
            metrics5 = model5.evaluate(X_test_5, y_test_5)
            
            p5 = os.path.join(artifacts_dir, "model_5_risk.pkl")
            save_model(model5, p5)
            saved_files.append(p5)
            results["model_5"] = metrics5
            
            accuracy = metrics5.get('accuracy', 'N/A')
            if log_callback:
                log_callback(f"✅ Model 5 complete — Accuracy: {accuracy}")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 5 failed: {str(e)}", level="error")
            results["model_5"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 6: Goal Achievement ==========
        step = 6
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Goal Achievement")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Goal Achievement Model...")
            
            X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(
                X_monthly, y_goal, test_size=test_size, random_state=seed
            )
            model6 = GoalAchievementModel(n_estimators=100, max_depth=7)
            model6.train(X_train_6, y_train_6)
            metrics6 = model6.evaluate(X_test_6, y_test_6)
            
            p6 = os.path.join(artifacts_dir, "model_6_goal.pkl")
            save_model(model6, p6)
            saved_files.append(p6)
            results["model_6"] = metrics6
            
            accuracy = metrics6.get('accuracy', 'N/A')
            if log_callback:
                log_callback(f"✅ Model 6 complete — Accuracy: {accuracy}")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 6 failed: {str(e)}", level="error")
            results["model_6"] = {"error": str(e), "trace": traceback.format_exc()}
        
        # ========== MODEL 7: Churn Prediction ==========
        step = 7
        if progress_callback:
            progress_callback(step, total_steps, f"[{step}/7] Training Churn Prediction")
        try:
            if log_callback:
                log_callback(f"[{step}/7] Training Churn Prediction Model...")
            
            X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(
                X_monthly, y_goal, test_size=test_size, random_state=seed
            )
            model7 = ChurnPredictionModel(n_estimators=100)
            model7.train(X_train_7, y_train_7)
            metrics7 = model7.evaluate(X_test_7, y_test_7)
            
            p7 = os.path.join(artifacts_dir, "model_7_churn.pkl")
            save_model(model7, p7)
            saved_files.append(p7)
            results["model_7"] = metrics7
            
            accuracy = metrics7.get('accuracy', 'N/A')
            if log_callback:
                log_callback(f"✅ Model 7 complete — Accuracy: {accuracy}")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model 7 failed: {str(e)}", level="error")
            results["model_7"] = {"error": str(e), "trace": traceback.format_exc()}
        
        if log_callback:
            log_callback("🎉 All models trained successfully!", level="info")
    
    except Exception as e:
        if log_callback:
            log_callback(f"💥 Pipeline error: {str(e)}", level="error")
        results["_pipeline_error"] = str(e)
    
    return results, saved_files

# ---- Training Execution ----
if run_button:
    if df_preview is None:
        st.error("❌ No data loaded. Upload a Parquet file or verify the local file exists.")
    else:
        # Reset session state
        st.session_state.logs = []
        st.session_state.metrics = {}
        st.session_state.trained_files = []
        
        # UI elements for progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        def progress_cb(step, total, message=None):
            """Update progress bar"""
            frac = min(100, int((step / total) * 100))
            progress_bar.progress(frac)
            if message:
                status_text.info(message)
        
        def log_cb(message, level="info"):
            """Add message to logs"""
            log(message, level=level)
        
        try:
            with st.spinner("⏳ Training models... This may take several minutes"):
                metrics, saved_files = run_training_pipeline(
                    df_preview, 
                    artifacts_dir=artifacts_dir, 
                    seed=int(random_seed),
                    test_size=test_size,
                    progress_callback=progress_cb, 
                    log_callback=log_cb
                )
            
            # Mark training complete
            progress_bar.progress(100)
            st.session_state.metrics = metrics
            st.session_state.trained_files = saved_files
            
            st.success("✅ Training completed successfully!")
            
            # Display Results Section
            st.subheader("📈 Model Metrics & Results")
            for model_name, metrics_dict in metrics.items():
                if isinstance(metrics_dict, dict) and model_name != "_pipeline_error":
                    with st.expander(f"**{model_name}**", expanded=False):
                        st.json(metrics_dict)
            
            # Display Saved Files Section
            st.subheader("💾 Saved Model Artifacts")
            if saved_files:
                for file_path in saved_files:
                    try:
                        file_size = os.path.getsize(file_path)
                        st.write(f"✅ `{os.path.basename(file_path)}` — {file_size:,} bytes")
                    except Exception as e:
                        st.write(f"⚠️ `{os.path.basename(file_path)}` — Could not determine size")
                
                # Create ZIP download
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in saved_files:
                        arcname = os.path.basename(fp)
                        zf.write(fp, arcname)
                buf.seek(0)
                
                st.download_button(
                    label="⬇️ Download All Models (ZIP)",
                    data=buf,
                    file_name="models_artifacts.zip",
                    mime="application/zip"
                )
            else:
                st.warning("⚠️ No models were saved. Check logs for errors.")
        
        except MemoryError:
            st.error("❌ Out of memory! Dataset is too large for this instance.")
            st.info("Try with a smaller dataset or upgrade your Streamlit tier.")
        except Exception as e:
            st.error("❌ Training pipeline failed with an exception")
            st.exception(e)
            st.subheader("📋 Recent Logs")
            st.code("\n".join(st.session_state.logs[-50:]))

# ---- Logs Viewer (Always Visible) ----
st.subheader("📋 Training Logs")
log_text = "\n".join(st.session_state.logs[-500:]) if st.session_state.logs else "No logs yet..."
st.text_area(
    "View all training logs", 
    value=log_text, 
    height=300, 
    disabled=True,
    key="logs_viewer"
)
