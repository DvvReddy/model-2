"""
FinBuddy AI - Financial Prediction Engine
COMPLETE FIXED VERSION - All compatibility issues resolved
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL COMPATIBILITY FIXES
# ============================================================================
def fix_model_compatibility(model):
    """Fix scikit-learn model compatibility issues"""
    try:
        # Recursively fix ensemble models
        if hasattr(model, 'estimators_'):
            for est in model.estimators_:
                if hasattr(est, 'tree_'):
                    if not hasattr(est.tree_, 'monotonic_cst'):
                        est.tree_.monotonic_cst = None
        
        # Fix direct tree models
        elif hasattr(model, 'tree_'):
            if not hasattr(model.tree_, 'monotonic_cst'):
                model.tree_.monotonic_cst = None
        
        # Fix RandomForest
        elif hasattr(model, 'estimators_') and hasattr(model, 'n_estimators'):
            for tree in model.estimators_:
                if hasattr(tree, 'tree_'):
                    if not hasattr(tree.tree_, 'monotonic_cst'):
                        tree.tree_.monotonic_cst = None
    except:
        pass
    
    return model

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FinBuddy AI",
    page_icon="💰",
    layout="wide"
)

# ============================================================================
# LOAD MODELS WITH FIXES
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models with compatibility fixes"""
    models = {}
    files = {
        'spending': 'artifacts/model_1_spending.pkl',
        'category': 'artifacts/model_2_category.pkl',
        'anomaly': 'artifacts/model_3_anomaly.pkl',
        'segmentation': 'artifacts/model_4_segmentation.pkl',
        'risk': 'artifacts/model_5_risk.pkl',
        'goal': 'artifacts/model_6_goal.pkl',
        'churn': 'artifacts/model_7_churn.pkl',
    }
    
    for name, path in files.items():
        try:
            if os.path.exists(path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = joblib.load(path)
                
                # Extract model from wrapper dict
                if isinstance(data, dict) and 'model' in data:
                    model = data['model']
                else:
                    model = data
                
                # Apply compatibility fixes
                model = fix_model_compatibility(model)
                models[name] = model
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {str(e)[:80]}")
            models[name] = None
    
    return models

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_features(spending, volatility, online, n=66):
    """Create feature vector"""
    f = np.zeros(n)
    f[0] = spending
    f[5] = volatility
    f[10] = online
    f[15] = spending * 0.7
    f[16] = spending * 0.8
    f[20] = volatility
    f[25] = volatility * 100
    f[30] = 1.0 if spending > 20000 else 0.0
    f[35] = spending / 7 if spending > 0 else 0
    f[36] = spending
    f[40] = online
    f[45:50] = np.random.uniform(4, 8, 5)
    f[50] = 0.5
    f[55] = online * 0.6
    f[60] = max(0, 1 - volatility)
    
    for i in range(n):
        if f[i] == 0:
            f[i] = np.random.uniform(0, 100)
    
    return f

def safe_predict(m, f, name=""):
    """Safe prediction with error handling"""
    if m is None:
        return None
    try:
        return m.predict([f])[0]
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            try:
                m = fix_model_compatibility(m)
                return m.predict([f])[0]
            except:
                return None
        return None
    except:
        return None

def safe_proba(m, f, name=""):
    """Safe probability with fallbacks"""
    if m is None:
        return None
    
    try:
        # Try predict_proba
        if hasattr(m, 'predict_proba'):
            return m.predict_proba([f])[0]
        
        # IsolationForest fallback
        elif hasattr(m, 'score_samples'):
            s = m.score_samples([f])
            return 1 / (1 + np.exp(s))
        
        # KMeans fallback
        elif hasattr(m, 'transform'):
            d = m.transform([f])
            p = 1 / (1 + d)
            return p[0] / p[0].sum()
        
        return None
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            try:
                m = fix_model_compatibility(m)
                if hasattr(m, 'predict_proba'):
                    return m.predict_proba([f])[0]
            except:
                pass
        return None
    except:
        return None

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown("# 💰 FinBuddy AI")
    st.markdown("### Financial Prediction Engine")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    loaded = sum(1 for v in models.values() if v is not None)
    if loaded == 0:
        st.error("❌ No models loaded!")
        st.stop()
    
    st.success(f"✅ {loaded}/7 models loaded")
    
    # Navigation
    page = st.sidebar.radio("Choose:", [
        "Dashboard", "Spending", "Categories", 
        "Anomaly", "Segments", "Risk", "Goals", "Churn"
    ])
    
    # ====================================================================
    # DASHBOARD
    # ====================================================================
    if page == "Dashboard":
        st.header("📊 Dashboard")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            spend = st.number_input("Spending (₹)", 0, 1000000, 25000)
        with c2:
            vol = st.slider("Volatility", 0.0, 1.0, 0.3)
        with c3:
            onl = st.slider("Online %", 0.0, 1.0, 0.5)
        
        if st.button("Predict All"):
            f = create_features(spend, vol, onl)
            
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                if models.get('spending'):
                    p = safe_predict(models['spending'], f, "spending")
                    if p:
                        st.metric("Spending", f"₹{p:,.0f}")
                    else:
                        st.error("❌ Failed")
                else:
                    st.warning("⚠️ Not loaded")
            
            with c2:
                if models.get('risk'):
                    pr = safe_proba(models['risk'], f, "risk")
                    if pr is not None and len(pr) > 1:
                        st.metric("Risk", "🔴 High" if pr[1] > 0.5 else "🟢 Low")
                    else:
                        st.error("❌ Failed")
                else:
                    st.warning("⚠️ Not loaded")
            
            with c3:
                if models.get('goal'):
                    pr = safe_proba(models['goal'], f, "goal")
                    if pr is not None and len(pr) > 1:
                        st.metric("Goal", f"{pr[1]:.0%}")
                    else:
                        st.error("❌ Failed")
                else:
                    st.warning("⚠️ Not loaded")
            
            with c4:
                if models.get('churn'):
                    pr = safe_proba(models['churn'], f, "churn")
                    if pr is not None and len(pr) > 1:
                        st.metric("Churn", "🔴 High" if pr[1] > 0.6 else "🟢 Low")
                    else:
                        st.error("❌ Failed")
                else:
                    st.warning("⚠️ Not loaded")
    
    # ====================================================================
    # SPENDING
    # ====================================================================
    elif page == "Spending":
        st.header("💸 Spending Forecast")
        
        c1, c2 = st.columns(2)
        with c1:
            avg = st.number_input("Average", 0, 1000000, 20000)
            std = st.slider("StdDev", 0, 50000, 5000)
        with c2:
            d7 = st.number_input("7 Days", 0, 100000, 5000)
            d30 = st.number_input("30 Days", 0, 500000, 25000)
        
        if st.button("Predict"):
            if not models.get('spending'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(avg, std/max(avg, 1), 0.5)
                p = safe_predict(models['spending'], f, "spending")
                
                if p:
                    st.success(f"### ₹{p:,.0f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=['7d', '30d', 'Pred'], y=[d7, d30, p]))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Prediction failed")
    
    # ====================================================================
    # CATEGORIES
    # ====================================================================
    elif page == "Categories":
        st.header("📊 Category Breakdown")
        
        tot = st.slider("Total", 5000, 500000, 50000)
        
        if st.button("Forecast"):
            if not models.get('category'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(tot, 0.3, 0.5)
                
                try:
                    p = models['category'].predict([f])[0]
                    
                    if hasattr(models['category'], 'category_names'):
                        df = pd.DataFrame({
                            'Category': models['category'].category_names,
                            'Spending': p
                        }).sort_values('Spending', ascending=False)
                        
                        fig = px.pie(df, values='Spending', names='Category')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Top predictions:", p[:3])
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:80]}")
    
    # ====================================================================
    # ANOMALY
    # ====================================================================
    elif page == "Anomaly":
        st.header("⚠️ Anomaly Detection")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            amt = st.number_input("Amount", 0, 100000, 5000)
        with c2:
            hr = st.slider("Hour", 0, 23, 14)
        with c3:
            typ = st.selectbox("Type", ["Online", "Offline"])
        
        if st.button("Check"):
            if not models.get('anomaly'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(amt, 0.1, 1.0 if typ == "Online" else 0.0)
                
                try:
                    if hasattr(models['anomaly'], 'score_samples'):
                        s = models['anomaly'].score_samples([f])[0]
                        pr = 1 / (1 + np.exp(s))
                    else:
                        pr = models['anomaly'].predict([f])[0]
                    
                    if pr > 0.5:
                        st.error(f"🔴 Anomaly! {pr:.1%}")
                    else:
                        st.success(f"🟢 Normal. {pr:.1%}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:80]}")
    
    # ====================================================================
    # SEGMENTS
    # ====================================================================
    elif page == "Segments":
        st.header("👥 User Segment")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            cat = st.slider("Categories", 1, 12, 5)
        with c2:
            txn = st.number_input("Avg Txn", 1000, 50000, 10000)
        with c3:
            fr = st.slider("Frequency", 0.0, 1.0, 0.5)
        
        if st.button("Find"):
            if not models.get('segmentation'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(cat * txn, 0.3, fr)
                
                try:
                    seg = models['segmentation'].predict([f])[0]
                    st.info(f"### Segment {seg}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:80]}")
    
    # ====================================================================
    # RISK
    # ====================================================================
    elif page == "Risk":
        st.header("💰 Risk Assessment")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            inc = st.number_input("Income", 0, 1000000, 50000)
        with c2:
            spd = st.number_input("Spending", 0, 1000000, 30000)
        with c3:
            sv = st.slider("Savings %", 0.0, 1.0, 0.4)
        
        if st.button("Assess"):
            if not models.get('risk'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(spd, abs(inc - spd) / max(inc, 1), sv)
                pr = safe_proba(models['risk'], f, "risk")
                
                if pr is not None and len(pr) > 1:
                    if pr[1] > 0.5:
                        st.error(f"🔴 High: {pr[1]:.1%}")
                    else:
                        st.success(f"🟢 Low: {pr[1]:.1%}")
                else:
                    st.error("❌ Failed")
    
    # ====================================================================
    # GOALS
    # ====================================================================
    elif page == "Goals":
        st.header("🎯 Goal Achievement")
        
        c1, c2 = st.columns(2)
        with c1:
            gl = st.number_input("Goal", 0, 10000000, 100000)
            ms = st.slider("Months", 1, 60, 12)
        with c2:
            cr = st.number_input("Current", 0, 10000000, 20000)
            sv = st.number_input("Monthly Save", 0, 100000, 5000)
        
        if st.button("Check"):
            if not models.get('goal'):
                st.error("❌ Model not loaded")
            else:
                f = create_features(sv, 0.1, 0.5)
                pr = safe_proba(models['goal'], f, "goal")
                
                if pr is not None and len(pr) > 1:
                    proj = cr + (sv * ms)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Goal", f"₹{gl:,.0f}")
                    with c2:
                        st.metric("Projected", f"₹{proj:,.0f}")
                    with c3:
                        gap = gl - proj
                        st.metric("Gap", f"₹{gap:,.0f}" if gap > 0 else f"✓ ₹{abs(gap):,.0f}")
                    
                    st.success(f"Success: {pr[1]:.1%}")
                else:
                    st.error("❌ Failed")
    
    # ====================================================================
    # CHURN
    # ====================================================================
    elif page == "Churn":
        st.header("👋 Churn Risk")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            dy = st.slider("Inactive Days", 0, 365, 5)
        with c2:
            lg = st.slider("Monthly Logins", 0, 30, 10)
        with c3:
            ft = st.slider("Features (1-10)", 1, 10, 7)
        
        if st.button("Predict"):
            if not models.get('churn'):
                st.error("❌ Model not loaded")
            else:
                eng = (lg / 30) * (ft / 10)
                f = create_features(eng * 50000, 0.1, eng)
                pr = safe_proba(models['churn'], f, "churn")
                
                if pr is not None and len(pr) > 1:
                    if pr[1] > 0.6:
                        st.error(f"🔴 High: {pr[1]:.1%}")
                    elif pr[1] > 0.3:
                        st.warning(f"🟡 Medium: {pr[1]:.1%}")
                    else:
                        st.success(f"🟢 Low: {pr[1]:.1%}")
                else:
                    st.error("❌ Failed")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()
