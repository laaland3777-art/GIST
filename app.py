import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ==========================================
# 页面设置
# ==========================================
st.set_page_config(
    page_title="Risk Probability Predictor",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# 加载模型和配置
# ==========================================
@st.cache_resource
def load_models():
    logistic_model = joblib.load("deployment_model/logistic_model.pkl")
    rf_model = joblib.load("deployment_model/rf_model.pkl")
    catboost_model = joblib.load("deployment_model/catboost_model.pkl")

    with open("deployment_model/model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return logistic_model, rf_model, catboost_model, config

logistic_model, rf_model, catboost_model, config = load_models()

# ==========================================
# Soft Voting 概率融合
# ==========================================
def soft_voting_predict_proba(input_df, weights):
    p1 = logistic_model.predict_proba(input_df)[:, 1]
    p2 = rf_model.predict_proba(input_df)[:, 1]
    p3 = catboost_model.predict_proba(input_df)[:, 1]

    weight_array = np.array([
        weights["logistic"],
        weights["rf"],
        weights["catboost"]
    ], dtype=float)

    weight_array = weight_array / weight_array.sum()

    probas = np.vstack([p1, p2, p3])
    final_proba = np.average(probas, axis=0, weights=weight_array)

    return final_proba

# ==========================================
# 页面标题
# ==========================================
st.title("🩺 Risk Probability Predictor")
st.markdown("### Online prediction of Risk probability using the soft voting ensemble model")

st.info(
    """
    **Variable definition**
    - **Growth pattern**: Endoluminal / Exophytic / Mixed
    - **Ulcer**: No / Yes
    - **Numeric variables**: Length, TG/HDL, SII, LFF, VFA
    """
)

# ==========================================
# 输入区域
# ==========================================
st.subheader("Input variables")

col1, col2 = st.columns(2)

with col1:
    growth_pattern_label = st.selectbox(
        "Growth pattern",
        options=list(config["growth_pattern_mapping"].keys())
    )

    length = st.number_input(
        "Length",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.4f"
    )

    sii = st.number_input(
        "SII",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.4f"
    )

    vfa = st.number_input(
        "VFA",
        min_value=0.0,
        value=50.0,
        step=1.0,
        format="%.4f"
    )

with col2:
    ulcer_label = st.selectbox(
        "Ulcer",
        options=list(config["ulcer_mapping"].keys())
    )

    tg_hdl = st.number_input(
        "TG/HDL",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.4f"
    )

    lff = st.number_input(
        "LFF",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.4f"
    )

# ==========================================
# 预测按钮
# ==========================================
if st.button("Predict Risk Probability", use_container_width=True):
    growth_pattern_value = config["growth_pattern_mapping"][growth_pattern_label]
    ulcer_value = config["ulcer_mapping"][ulcer_label]

    input_df = pd.DataFrame([{
        "Growth pattern": growth_pattern_value,
        "Ulcer": ulcer_value,
        "Length": length,
        "TG/HDL": tg_hdl,
        "SII": sii,
        "LFF": lff,
        "VFA": vfa
    }])

    risk_proba = soft_voting_predict_proba(input_df, config["weights"])[0]
    pred_class = int(risk_proba >= 0.5)

    st.markdown("## Prediction Result")
    st.metric("Risk Probability", f"{risk_proba:.4f}")
    st.metric("Risk Percentage", f"{risk_proba * 100:.2f}%")

    progress_value = int(round(risk_proba * 100))
    st.progress(progress_value)

    if pred_class == 1:
        st.error("Prediction: High Risk")
    else:
        st.success("Prediction: Low Risk")

    with st.expander("View input data"):
        st.dataframe(input_df, use_container_width=True)
