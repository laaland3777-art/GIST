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

st.title("🩺 Risk Probability Predictor")
st.markdown("输入患者变量，在线预测 **Risk 的概率**")

# ==========================================
# 加载模型
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("soft_voting_model.pkl")
    return model

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
# 用户输入区域
# ==========================================
st.subheader("Please enter the variables")

growth_pattern_label = st.selectbox(
    "Growth pattern",
    options=["Endoluminal", "Exophytic", "Mixed"]
)

ulcer_label = st.selectbox(
    "Ulcer",
    options=["No", "Yes"]
)

length = st.number_input("Length", min_value=0.0, value=1.0, step=0.1, format="%.4f")
tg_hdl = st.number_input("TG/HDL", min_value=0.0, value=1.0, step=0.1, format="%.4f")
sii = st.number_input("SII", min_value=0.0, value=100.0, step=1.0, format="%.4f")
lff = st.number_input("LFF", min_value=0.0, value=1.0, step=0.1, format="%.4f")
vfa = st.number_input("VFA", min_value=0.0, value=50.0, step=1.0, format="%.4f")

# ==========================================
# 预测按钮
# ==========================================
if st.button("Predict Risk Probability"):
    growth_pattern_value = growth_pattern_map[growth_pattern_label]
    ulcer_value = ulcer_map[ulcer_label]

    input_df = pd.DataFrame([{
        "Growth pattern": growth_pattern_value,
        "Ulcer": ulcer_value,
        "Length": length,
        "TG/HDL": tg_hdl,
        "SII": sii,
        "LFF": lff,
        "VFA": vfa
    }])

    pred_proba = model.predict_proba(input_df)[0, 1]
    pred_class = int(pred_proba >= 0.5)

    st.markdown("## Prediction Result")
    st.write(f"**Risk Probability:** {pred_proba:.4f}")
    st.write(f"**Risk Percentage:** {pred_proba * 100:.2f}%")

    if pred_class == 1:
        st.error("Predicted Class: High Risk")
    else:
        st.success("Predicted Class: Low Risk")

    with st.expander("Input values used for prediction"):
        st.dataframe(input_df)
