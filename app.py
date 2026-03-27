import streamlit as st
import pandas as pd
import joblib

# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="Risk Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Risk Prediction Web App")
st.markdown("请输入患者特征，点击按钮后获得风险预测结果。")

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("ensemble_model_pipeline.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_artifacts()

# =========================
# 输入区域
# =========================
st.subheader("请输入特征")

growth_pattern = st.selectbox(
    "Grouth pattern",
    options=["1", "2", "3"]
)

ulcer = st.selectbox(
    "Ulcer",
    options=["0", "1"]
)

length = st.number_input("Length", min_value=0.0, value=5.0, step=0.1)
tg_hdl = st.number_input("TG/HDL", min_value=0.0, value=1.0, step=0.1)
sii = st.number_input("SII", min_value=0.0, value=500.0, step=1.0)
lff = st.number_input("LFF", min_value=0.0, value=10.0, step=0.1)
vfa = st.number_input("VFA", min_value=0.0, value=50.0, step=0.1)

# =========================
# 构造输入数据
# =========================
input_df = pd.DataFrame([{
    "Grouth pattern": growth_pattern,
    "Ulcer": ulcer,
    "Length": length,
    "TG/HDL": tg_hdl,
    "SII": sii,
    "LFF": lff,
    "VFA": vfa
}])

st.markdown("### 当前输入数据")
st.dataframe(input_df)

# =========================
# 预测
# =========================
if st.button("开始预测"):
    try:
        pred_prob = model.predict_proba(input_df)[0, 1]
        pred_class = model.predict(input_df)[0]

        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform([pred_class])[0]
        else:
            pred_label = pred_class

        st.success("预测完成！")
        st.markdown(f"### 预测结果：`{pred_label}`")
        st.markdown(f"### 风险概率：`{pred_prob:.4f}`")

        if pred_prob >= 0.5:
            st.error("模型判断：较高风险")
        else:
            st.info("模型判断：较低风险")

    except Exception as e:
        st.error(f"预测失败：{e}")
