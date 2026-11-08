"""Streamlit dashboard for the Credit Default Risk Analysis project."""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis import load_data, build_pipeline

st.set_page_config(page_title="Credit Default Risk Analysis", page_icon="📊", layout="wide")
st.title("Credit Default Risk Analysis")
st.caption("A junior data-science portfolio project by Brian Owino Abayo | UCI Taiwan credit-card dataset")

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model_and_results():
    data = get_data()
    X = data.drop(columns=["default_next_month"])
    y = data["default_next_month"]
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.50).astype(int)
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0)
    }
    results = X_test.copy()
    results["actual_default"] = y_test.values
    results["risk_score"] = probs
    return data, model, metrics, results

try:
    data, model, metrics, results = get_model_and_results()
except Exception as e:
    st.error("The dataset could not be loaded. Check your internet connection and install requirements.txt.")
    st.exception(e)
    st.stop()

n = len(data)
rate = data["default_next_month"].mean()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Records", f"{n:,}")
c2.metric("Default rate", f"{rate:.1%}")
c3.metric("Model ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
c4.metric("Test records", f"{len(results):,}")

st.info("Dataset context: 30,000 Taiwan credit-card client records. These results are for learning and are not evidence of performance on Kenyan microfinance borrowers.")

tab1, tab2, tab3, tab4 = st.tabs(["Portfolio overview", "Explore the data", "Model performance", "Risk score demo"])

with tab1:
    st.subheader("What the analysis covers")
    st.markdown("""
    - Data quality checks and exploratory analysis
    - Patterns in repayment status, credit limits and payment behaviour
    - Logistic-regression baseline with stratified hold-out evaluation
    - Precision/recall trade-offs and threshold exploration
    """)
    counts = data["default_next_month"].value_counts().rename(index={0:"No default",1:"Default"}).reset_index()
    counts.columns = ["Outcome","Clients"]
    fig = px.bar(counts, x="Outcome", y="Clients", color="Outcome", title="Next-month repayment outcome")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The outcome distribution is imbalanced; accuracy alone can hide poor detection of the minority class.")

with tab2:
    st.subheader("Explore the data")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    default_col = "default_next_month"
    candidates = [c for c in numeric_cols if c != default_col]
    selected = st.selectbox("Choose a numeric variable", candidates, index=0 if candidates else None)
    if selected:
        fig = px.histogram(data, x=selected, color=default_col, barmode="overlay",
                           histnorm="probability", opacity=0.65,
                           labels={default_col:"Next-month default (0=no, 1=yes)"},
                           title=f"Distribution of {selected} by outcome")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Summary by outcome**")
        st.dataframe(data.groupby(default_col)[selected].agg(["count","mean","median","std"]).rename(index={0:"No default",1:"Default"}), use_container_width=True)
    st.markdown("**Sample records**")
    st.dataframe(data.head(15), use_container_width=True)

with tab3:
    st.subheader("Baseline model evaluation")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:.3f}")
    st.markdown("Metrics shown on a held-out test set at a 0.50 classification threshold, except ROC-AUC which uses predicted probabilities.")
    from sklearn.metrics import confusion_matrix, roc_curve
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    pred = (results["risk_score"] >= threshold).astype(int)
    cm = confusion_matrix(results["actual_default"], pred)
    fig = px.imshow(cm, text_auto=True, x=["Predicted no default","Predicted default"],
                    y=["Actual no default","Actual default"], color_continuous_scale="Blues",
                    title=f"Confusion matrix at threshold {threshold:.2f}")
    st.plotly_chart(fig, use_container_width=True)
    from sklearn.metrics import precision_score, recall_score, f1_score
    m1,m2,m3 = st.columns(3)
    m1.metric("Precision", f"{precision_score(results.actual_default,pred,zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(results.actual_default,pred,zero_division=0):.3f}")
    m3.metric("F1-score", f"{f1_score(results.actual_default,pred,zero_division=0):.3f}")
    st.caption("Lowering the threshold generally identifies more potential defaults (higher recall) but can also increase false positives. Threshold selection depends on business costs and must be validated with local data.")

with tab4:
    st.subheader("Illustrative score inspection")
    st.warning("This is a demonstration using records from the public test dataset, not a real applicant assessment or lending recommendation.")
    display = results[["risk_score","actual_default"]].copy()
    display["actual_default"] = display["actual_default"].map({0:"No default",1:"Default"})
    display["risk_band"] = pd.cut(display["risk_score"], bins=[-0.001,0.25,0.50,0.75,1.001],
                                  labels=["Lower model score","Moderate model score","Elevated model score","Higher model score"])
    band = display["risk_band"].value_counts().sort_index().reset_index()
    band.columns=["Model score band","Test records"]
    st.plotly_chart(px.bar(band,x="Model score band",y="Test records",title="Test-set records by model score band"), use_container_width=True)
    st.dataframe(display.sample(min(20,len(display)), random_state=42), use_container_width=True)

with st.expander("Methodology and responsible-use notes"):
    st.markdown("""
    **Model:** Logistic regression with class weighting. Numeric variables are median-imputed and standardised; categorical variables are imputed and one-hot encoded.

    **Validation:** Stratified 75/25 train/test split, random state 42. The test set is held out from fitting.

    **Limitations:** This historical Taiwan credit-card dataset is not Kenyan lending data. No model from this exercise should be used for real credit decisions. Before any real-world use, a lender would need representative local data, legal and privacy review, fairness and bias testing, calibration, independent validation, monitoring and human oversight.
    """)
