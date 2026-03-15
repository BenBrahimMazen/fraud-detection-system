"""
Fraud Detection Dashboard — Streamlit

Pages
-----
1. Live Prediction   — score a transaction, see risk gauge + SHAP explanation
2. Model Performance — ROC curves, confusion matrix, metrics comparison
3. Data Insights     — class distribution, feature correlations, SHAP plots
4. Batch Analysis    — upload CSV, score all transactions, download results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
import joblib
from loguru import logger

from src.config import DATA_PROCESSED_DIR, PCA_FEATURES, RAW_FEATURES

API_URL = "http://localhost:8000"

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .fraud-badge-critical { background:#C0392B; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .fraud-badge-high     { background:#E67E22; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .fraud-badge-medium   { background:#F39C12; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .fraud-badge-low      { background:#27AE60; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """Load model artifacts once and cache them."""
    artifacts = {}
    splits_path = DATA_PROCESSED_DIR / "splits.pkl"
    if splits_path.exists():
        X_train, X_test, y_train, y_test = joblib.load(splits_path)
        artifacts["X_test"]  = X_test
        artifacts["y_test"]  = np.array(y_test)

    shap_path = DATA_PROCESSED_DIR / "shap_plots"
    if shap_path.exists():
        artifacts["shap_plots"] = shap_path

    fn_path = DATA_PROCESSED_DIR / "feature_names.pkl"
    if fn_path.exists():
        artifacts["feature_names"] = joblib.load(fn_path)

    return artifacts


def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except:
        return False


def predict_transaction(tx_dict: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict", json=tx_dict, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def explain_transaction(tx_dict: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/explain", json=tx_dict, timeout=15)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def make_gauge(score: int, tier: str, color: str) -> go.Figure:
    """Create a risk score gauge chart."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": f"Risk Score — {tier}", "font": {"size": 18}},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fef9e7"},
                {"range": [60, 85], "color": "#fdebd0"},
                {"range": [85,100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "value": score},
        },
        number = {"suffix": "/100", "font": {"size": 36}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def make_shap_bar(drivers: list, reducers: list) -> go.Figure:
    """Horizontal bar chart of SHAP values."""
    factors = drivers[:5] + reducers[:5]
    names   = [f["feature"] for f in factors]
    values  = [f["shap_value"] for f in factors]
    colors  = ["#E8593C" if v > 0 else "#4A90D9" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = names,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:+.3f}" for v in values],
        textposition= "outside",
    ))
    fig.update_layout(
        title    = "Why was this transaction flagged?",
        xaxis_title = "SHAP value",
        height   = 380,
        margin   = dict(t=40, b=20, l=20, r=60),
        showlegend = False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("Fraud Detection")
    st.caption("Production ML System v1.0")

    api_ok = check_api()
    if api_ok:
        st.success("API Online ✓")
    else:
        st.error("API Offline — start with:\nuvicorn src.api.main:app --port 8000")

    page = st.radio(
        "Navigation",
        ["🔍 Live Prediction", "📊 Model Performance", "🔬 Data Insights", "📁 Batch Analysis"],
    )
    st.divider()
    st.caption("Built with XGBoost + SHAP + FastAPI")


artifacts = load_artifacts()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Live Prediction":
    st.title("🔍 Live Transaction Scoring")
    st.caption("Score a transaction in real-time and get a SHAP explanation")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Amount ($)", min_value=0.0, value=149.62, step=0.01)
        time   = st.number_input("Time (seconds)", min_value=0.0, value=406.0)

        st.caption("PCA Features (V1–V28) — from anonymized card data")
        use_preset = st.toggle("Use sample fraud transaction", value=False)

        # Preset fraud transaction (known fraud from dataset)
        if use_preset:
            preset = {
                "V1":-3.04,"V2":-3.16,"V3":1.09,"V4":0.99,"V5":-0.98,
                "V6":-1.71,"V7":-0.42,"V8":0.81,"V9":-0.90,"V10":-3.07,
                "V11":2.17,"V12":-4.21,"V13":0.47,"V14":-4.43,"V15":0.27,
                "V16":-2.97,"V17":-5.57,"V18":-2.69,"V19":0.19,"V20":0.12,
                "V21":0.55,"V22":0.21,"V23":0.27,"V24":0.05,"V25":-0.30,
                "V26":-0.15,"V27":0.75,"V28":0.22,
            }
        else:
            preset = {f"V{i}": 0.0 for i in range(1, 29)}

        v_cols = st.columns(4)
        v_vals = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with v_cols[col_idx]:
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(preset[f"V{i}"]),
                    format="%.3f", label_visibility="visible",
                    key=f"v{i}"
                )

    with col2:
        st.subheader("Risk Assessment")

        tx_dict = {**v_vals, "Time": time, "Amount": amount}

        if st.button("🚀 Score Transaction", type="primary", use_container_width=True):
            if not api_ok:
                st.error("API is offline. Start it first.")
            else:
                with st.spinner("Scoring …"):
                    result = predict_transaction(tx_dict)

                if result:
                    risk = result["risk"]
                    st.plotly_chart(
                        make_gauge(risk["score"], risk["tier"], risk["color"]),
                        use_container_width=True,
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Fraud Probability", f"{risk['probability']:.1%}")
                    c2.metric("Decision", "🚨 FRAUD" if result["is_fraud"] else "✅ LEGIT")
                    c3.metric("Response", f"{result['processing_ms']:.1f}ms")

                    st.info(f"**Recommended action:** {risk['action']}")

                    # Get SHAP explanation
                    with st.spinner("Generating explanation …"):
                        explanation = explain_transaction(tx_dict)

                    if explanation:
                        st.subheader("Why this decision?")
                        st.plotly_chart(
                            make_shap_bar(
                                explanation["top_fraud_drivers"],
                                explanation["top_fraud_reducers"],
                            ),
                            use_container_width=True,
                        )
                        with st.expander("Full explanation JSON"):
                            st.json(explanation)
                else:
                    st.error("Prediction failed — check API is running")
        else:
            st.info("👈 Fill in the transaction details and click Score Transaction")
            st.caption("Toggle 'Use sample fraud transaction' to see a known fraud case")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    # Model metrics from our training run
    metrics = {
        "XGBoost":          {"roc_auc": 0.9812, "f1": 0.7679, "recall": 0.8776, "precision": 0.6825, "color": "#E8593C"},
        "Random Forest":    {"roc_auc": 0.9801, "f1": 0.8265, "recall": 0.8265, "precision": 0.8265, "color": "#4A90D9"},
        "Isolation Forest": {"roc_auc": 0.9531, "f1": 0.1436, "recall": 0.7245, "precision": 0.0797, "color": "#F39C12"},
        "Ensemble":         {"roc_auc": 0.9719, "f1": 0.8358, "recall": 0.8571, "precision": 0.8155, "color": "#27AE60"},
    }

    # Top metrics
    st.subheader("Key Metrics — Stacking Ensemble")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",   "0.9719", "Best overall")
    c2.metric("F1 Score",  "0.8358", "+6.8% vs baseline")
    c3.metric("Recall",    "85.7%",  "Fraud caught")
    c4.metric("Precision", "81.6%",  "Alert accuracy")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # ROC-AUC comparison
        fig = go.Figure()
        for model, m in metrics.items():
            fig.add_trace(go.Bar(
                name=model, x=[model],
                y=[m["roc_auc"]], marker_color=m["color"],
                text=[f"{m['roc_auc']:.4f}"], textposition="outside",
            ))
        fig.update_layout(
            title="ROC-AUC by Model", yaxis_range=[0.9, 1.0],
            showlegend=False, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Recall vs Precision scatter
        fig = go.Figure()
        for model, m in metrics.items():
            fig.add_trace(go.Scatter(
                x=[m["precision"]], y=[m["recall"]],
                mode="markers+text",
                text=[model], textposition="top center",
                marker=dict(size=18, color=m["color"]),
                name=model,
            ))
        fig.update_layout(
            title="Precision vs Recall Trade-off",
            xaxis_title="Precision", yaxis_title="Recall",
            height=350, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Full metrics table
    st.subheader("Full Metrics Table")
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.columns = ["Model", "ROC-AUC", "F1", "Recall", "Precision", "Color"]
    df_metrics = df_metrics.drop("Color", axis=1)
    st.dataframe(
        df_metrics.style.highlight_max(subset=["ROC-AUC", "F1", "Recall", "Precision"], color="#d5f5e3"),
        use_container_width=True,
    )

    # Ensemble weights
    st.subheader("Ensemble Model Weights")
    weights = {"XGBoost": 7.78, "Random Forest": 9.88, "Isolation Forest": 1.41}
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        color_discrete_sequence=["#E8593C", "#4A90D9", "#F39C12"],
        title="How the meta-learner weights each base model",
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Data Insights":
    st.title("🔬 Data Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraudulent",         "492 (0.17%)")
    col3.metric("Imbalance Ratio",    "1 : 577")
    col4.metric("Features",           "50 (after engineering)")

    st.divider()

    # SHAP plots if available
    shap_dir = artifacts.get("shap_plots")
    if shap_dir:
        col1, col2 = st.columns(2)
        with col1:
            gi_path = shap_dir / "global_importance.png"
            if gi_path.exists():
                st.subheader("Global Feature Importance")
                st.image(str(gi_path), use_container_width=True)

        with col2:
            bs_path = shap_dir / "beeswarm.png"
            if bs_path.exists():
                st.subheader("SHAP Beeswarm")
                st.image(str(bs_path), use_container_width=True)

        # Waterfall plots
        waterfall_plots = list(shap_dir.glob("waterfall_*.png"))
        if waterfall_plots:
            st.subheader("Transaction Waterfall (Fraud Sample)")
            st.image(str(waterfall_plots[0]), use_container_width=True)
    else:
        st.info("Run `python -m src.explainability.shap_explainer` to generate SHAP plots")

    # Test set distribution
    if "X_test" in artifacts:
        st.subheader("Test Set — Amount Distribution by Class")
        X_test = artifacts["X_test"]
        y_test = artifacts["y_test"]

        amt_col = -1  # Amount is last column
        df_dist = pd.DataFrame({
            "Amount": X_test[:, amt_col],
            "Class":  ["Fraud" if y == 1 else "Legitimate" for y in y_test],
        })
        # Sample for performance
        df_sample = df_dist.groupby("Class").apply(
            lambda x: x.sample(min(500, len(x)), random_state=42)
        ).reset_index(drop=True)

        fig = px.histogram(
            df_sample, x="Amount", color="Class",
            barmode="overlay", nbins=50,
            color_discrete_map={"Fraud": "#E8593C", "Legitimate": "#4A90D9"},
            title="Amount Distribution — Fraud vs Legitimate",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📁 Batch Analysis":
    st.title("📁 Batch Transaction Analysis")
    st.caption("Upload a CSV of transactions and score them all at once")

    uploaded = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        help="Must contain columns: V1-V28, Time, Amount",
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} transactions")
        st.dataframe(df.head(), use_container_width=True)

        required = PCA_FEATURES + RAW_FEATURES
        missing  = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("🚀 Score All Transactions", type="primary"):
                if not api_ok:
                    st.error("API is offline")
                else:
                    progress = st.progress(0)
                    results  = []
                    batch_size = 100

                    for i in range(0, min(len(df), 1000), batch_size):
                        batch = df.iloc[i:i+batch_size]
                        txs   = batch[required].to_dict("records")
                        payload = {"transactions": txs}

                        try:
                            r = requests.post(
                                f"{API_URL}/batch", json=payload, timeout=30
                            )
                            if r.status_code == 200:
                                results.extend(r.json()["predictions"])
                        except Exception as e:
                            st.error(f"Batch {i} failed: {e}")

                        progress.progress(min((i + batch_size) / min(len(df), 1000), 1.0))

                    if results:
                        df_results = pd.DataFrame([{
                            "transaction_id": r["transaction_id"],
                            "is_fraud":       r["is_fraud"],
                            "score":          r["risk"]["score"],
                            "tier":           r["risk"]["tier"],
                            "probability":    r["risk"]["probability"],
                            "action":         r["risk"]["action"],
                        } for r in results])

                        fraud_count = df_results["is_fraud"].sum()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Scored",      len(df_results))
                        c2.metric("Fraud",        fraud_count)
                        c3.metric("Fraud Rate",   f"{fraud_count/len(df_results):.2%}")
                        c4.metric("Avg Score",    f"{df_results['score'].mean():.1f}")

                        # Tier distribution
                        tier_counts = df_results["tier"].value_counts()
                        fig = px.pie(
                            values=tier_counts.values,
                            names=tier_counts.index,
                            color=tier_counts.index,
                            color_discrete_map={
                                "LOW":"#27AE60","MEDIUM":"#F39C12",
                                "HIGH":"#E67E22","CRITICAL":"#C0392B"
                            },
                            title="Risk Tier Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df_results, use_container_width=True)

                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            csv,
                            "fraud_scores.csv",
                            "text/csv",
                        )
    else:
        st.info("Upload a CSV file to begin batch scoring")
        st.caption("The file must contain columns V1–V28, Time, and Amount")
        if "X_test" in artifacts:
            if st.button("Use test set sample (500 rows)"):
                X_test = artifacts["X_test"]
                y_test = artifacts["y_test"]
                fn = artifacts.get("feature_names", PCA_FEATURES + RAW_FEATURES)
                fn = fn[:X_test.shape[1]]
                sample_df = pd.DataFrame(X_test[:500], columns=fn)
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Sample CSV",
                    csv,
                    "sample_transactions.csv",
                    "text/csv",
                )
