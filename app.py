import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 页面设置
# ==========================================
st.set_page_config(
    page_title="Risk Probability Predictor",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# 加载模型
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

model = load_model()

# ==========================================
# 分类变量映射
# ==========================================
growth_pattern_map = {
    "Endoluminal": 1,
    "Exophytic": 2,
    "Mixed": 3
}

ulcer_map = {
    "No": 0,
    "Yes": 1
}

# ==========================================
# 页面内容
# ==========================================
st.title("🩺 Risk Probability Predictor")
st.markdown("### 在线预测 Risk 概率")

st.info(
    """
    **变量说明**
    - **Growth pattern**: Endoluminal / Exophytic / Mixed
    - **Ulcer**: No / Yes
    - **连续变量**: Length, TG/HDL, SII, LFF, VFA
    """
)

st.subheader("请输入变量")

col1, col2 = st.columns(2)

with col1:
    growth_pattern_label = st.selectbox(
        "Growth pattern",
        options=["Endoluminal", "Exophytic", "Mixed"]
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
        options=["No", "Yes"]
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
# 预测
# ==========================================
if st.button("Predict Risk Probability", use_container_width=True):
    input_df = pd.DataFrame([{
        "Growth pattern": growth_pattern_map[growth_pattern_label],
        "Ulcer": ulcer_map[ulcer_label],
        "Length": length,
        "TG/HDL": tg_hdl,
        "SII": sii,
        "LFF": lff,
        "VFA": vfa
    }])

    risk_proba = model.predict_proba(input_df)[0, 1]
    pred_class = int(risk_proba >= 0.5)

    st.markdown("## Prediction Result")
    st.metric("Risk Probability", f"{risk_proba:.4f}")
    st.metric("Risk Percentage", f"{risk_proba * 100:.2f}%")
    st.progress(int(round(risk_proba * 100)))

    if pred_class == 1:
        st.error("Prediction: High Risk")
    else:
        st.success("Prediction: Low Risk")

    with st.expander("View input values"):
        st.dataframe(input_df, use_container_width=True)
