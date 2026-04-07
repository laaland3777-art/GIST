import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 页面配置
st.set_page_config(page_title="Risk 预测模型", layout="centered")

st.title("🩺 疾病 Risk 预测系统")
st.write("请输入患者的临床特征，系统将基于 Soft Voting 融合模型（CatBoost, Random Forest, Logistic Regression）预测 Risk 概率。")

# 加载模型 (使用缓存避免每次交互都重新加载)
@st.cache_resource
def load_models():
    return joblib.load('soft_voting_models.pkl')

try:
    models = load_models()
except FileNotFoundError:
    st.error("找不到模型文件 `soft_voting_models.pkl`，请确保文件已上传。")
    st.stop()

# ==========================================
# 构建网页输入界面
# ==========================================
st.subheader("📝 输入临床特征")

col1, col2 = st.columns(2)

with col1:
    # 分类变量输入
    growth_pattern_input = st.selectbox(
        "Growth pattern (生长模式)", 
        options=["Endoluminal", "Exophytic", "Mixed"]
    )
    
    ulcer_input = st.selectbox(
        "Ulcer (溃疡)", 
        options=["No", "Yes"]
    )
    
    # 连续变量输入
    length = st.number_input("Length", min_value=0.0, value=5.0, step=0.1)

with col2:
    tg_hdl = st.number_input("TG/HDL", min_value=0.0, value=1.0, step=0.1)
    sii = st.number_input("SII", min_value=0.0, value=500.0, step=10.0)
    lff = st.number_input("LFF", min_value=0.0, value=10.0, step=1.0)
    vfa = st.number_input("VFA", min_value=0.0, value=100.0, step=1.0)

# ==========================================
# 变量映射 (将网页选项映射回模型需要的数值)
# ==========================================
gp_mapping = {"Endoluminal": 1, "Exophytic": 2, "Mixed": 3}
ulcer_mapping = {"No": 0, "Yes": 1}

# ==========================================
# 预测逻辑
# ==========================================
if st.button("🚀 开始预测 Risk", type="primary"):
    # 构建输入 DataFrame
    input_data = pd.DataFrame({
        "Growth pattern": [gp_mapping[growth_pattern_input]],
        "Ulcer": [ulcer_mapping[ulcer_input]],
        "Length": [length],
        "TG/HDL": [tg_hdl],
        "SII": [sii],
        "LFF": [lff],
        "VFA": [vfa]
    })
    
    # Soft Voting 预测
    probas = []
    weights = [1, 1, 1] # 权重设置
    
    for name, model in models.items():
        # 获取预测为 1 的概率
        proba = model.predict_proba(input_data)[:, 1][0]
        probas.append(proba)
        
    # 计算加权平均概率
    final_proba = np.average(probas, weights=weights)
    
    st.markdown("---")
    st.subheader("📊 预测结果")
    
    # 结果展示
    if final_proba >= 0.5:
        st.error(f"**高风险 (High Risk)**")
    else:
        st.success(f"**低风险 (Low Risk)**")
        
    st.metric(label="Risk 发生概率", value=f"{final_proba * 100:.2f}%")
    
    # 展示进度条
    st.progress(float(final_proba))
    
    # 显示各个基模型的预测概率（可选，增加透明度）
    with st.expander("查看各基模型预测详情"):
        st.write(f"- **Logistic Regression**: {probas[0]*100:.2f}%")
        st.write(f"- **Random Forest**: {probas[1]*100:.2f}%")
        st.write(f"- **CatBoost**: {probas[2]*100:.2f}%")
