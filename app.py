import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.base import clone

warnings.filterwarnings("ignore")

# 页面配置
st.set_page_config(page_title="Risk 预测模型", layout="centered")

st.title("🩺 疾病 Risk 预测系统")
st.write("请输入患者的临床特征，系统将基于 Soft Voting 融合模型预测 Risk 概率。")

# ==========================================
# 核心改动：在云端动态训练模型，彻底告别 pkl 版本冲突！
# ==========================================
@st.cache_resource(show_spinner="正在初始化并训练模型，请稍候 (仅首次加载需要)...")
def train_and_get_models():
    try:
        # 1. 读取 GitHub 仓库里的数据文件
        df_train = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("❌ 找不到 `train.csv` 文件！请确保你已经将训练数据上传到了 GitHub 仓库中。")
        st.stop()

    target_col = "Risk"
    categorical_features = ["Growth pattern", "Ulcer"]
    numerical_features = ["Length", "TG/HDL", "SII", "LFF", "VFA"]
    selected_features = categorical_features + numerical_features

    df_train.dropna(subset=[target_col], inplace=True)
    X_train = df_train[selected_features].copy()
    y_train = df_train[target_col].copy()

    # 目标变量转成二分类
    def normalize_binary_target(y):
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = y.astype(str).str.strip()
            vals = list(sorted(y.unique()))
            mapping = {vals[0]: 0, vals[1]: 1}
            return y.map(mapping).astype(int)
        return y.astype(int)

    y_train = normalize_binary_target(y_train)

    # 2. 构建预处理管道
    numeric_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_preprocessor, numerical_features),
        ("cat", categorical_preprocessor, categorical_features)
    ])

    # 3. 定义模型
    logistic_best = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(C=0.01, solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42))
    ])
    rf_best = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_split=2, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    catboost_best = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", CatBoostClassifier(depth=6, iterations=100, learning_rate=0.1, verbose=0, random_state=42))
    ])

    # 4. 训练模型
    models = {
        "Logistic Regression": clone(logistic_best),
        "Random Forest": clone(rf_best),
        "CatBoost": clone(catboost_best)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

# 加载模型 (调用上面的函数)
models = train_and_get_models()

# ==========================================
# 构建网页输入界面
# ==========================================
st.markdown("---")
st.subheader("📝 输入临床特征")

col1, col2 = st.columns(2)

with col1:
    growth_pattern_input = st.selectbox("Growth pattern (生长模式)", options=["Endoluminal", "Exophytic", "Mixed"])
    ulcer_input = st.selectbox("Ulcer (溃疡)", options=["No", "Yes"])
    length = st.number_input("Length", min_value=0.0, value=5.0, step=0.1)

with col2:
    tg_hdl = st.number_input("TG/HDL", min_value=0.0, value=1.0, step=0.1)
    sii = st.number_input("SII", min_value=0.0, value=500.0, step=10.0)
    lff = st.number_input("LFF", min_value=0.0, value=10.0, step=1.0)
    vfa = st.number_input("VFA", min_value=0.0, value=100.0, step=1.0)

# 变量映射
gp_mapping = {"Endoluminal": 1, "Exophytic": 2, "Mixed": 3}
ulcer_mapping = {"No": 0, "Yes": 1}

# ==========================================
# 预测逻辑
# ==========================================
if st.button("🚀 开始预测 Risk", type="primary", use_container_width=True):
    # 构建输入 DataFrame (强制转换为正确的类型)
    input_data = pd.DataFrame({
        "Growth pattern": [int(gp_mapping[growth_pattern_input])],
        "Ulcer": [int(ulcer_mapping[ulcer_input])],
        "Length": [float(length)],
        "TG/HDL": [float(tg_hdl)],
        "SII": [float(sii)],
        "LFF": [float(lff)],
        "VFA": [float(vfa)]
    })
    
    # Soft Voting 预测
    probas = []
    
    for name, model in models.items():
        proba = model.predict_proba(input_data)[:, 1][0]
        probas.append(proba)
        
    # 计算加权平均概率 (权重均为1)
    final_proba = np.average(probas, weights=[1, 1, 1])
    
    st.markdown("---")
    st.subheader("📊 预测结果")
    
    # 结果展示
    if final_proba >= 0.5:
        st.error(f"**高风险 (High Risk)**")
    else:
        st.success(f"**低风险 (Low Risk)**")
        
    st.metric(label="Risk 发生概率", value=f"{final_proba * 100:.2f}%")
    st.progress(float(final_proba))
    
    # 显示各个基模型的预测概率
    with st.expander("查看各基模型预测详情"):
        st.write(f"- **Logistic Regression**: {probas[0]*100:.2f}%")
        st.write(f"- **Random Forest**: {probas[1]*100:.2f}%")
        st.write(f"- **CatBoost**: {probas[2]*100:.2f}%")
