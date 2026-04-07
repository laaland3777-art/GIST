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

# Page configuration
st.set_page_config(page_title="GIST Risk Prediction Model", layout="centered")

st.title("🩺 GIST Risk Prediction Model")
st.write("Please enter the patient's clinical features. The system will predict the risk probability based on a Soft Voting ensemble model.")

# ==========================================
# Core modification: Train models dynamically in the cloud to avoid .pkl version conflicts!
# ==========================================
@st.cache_resource(show_spinner="Initializing and training models, please wait (only required for the first load)...")
def train_and_get_models():
    try:
        # 1. Read the data file from the GitHub repository
        df_train = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("❌ `train.csv` file not found! Please ensure you have uploaded the training data to the GitHub repository.")
        st.stop()

    target_col = "Risk"
    categorical_features = ["Growth pattern", "Ulcer"]
    numerical_features = ["Length", "TG/HDL", "SII", "LFF", "VFA"]
    selected_features = categorical_features + numerical_features

    df_train.dropna(subset=[target_col], inplace=True)
    X_train = df_train[selected_features].copy()
    y_train = df_train[target_col].copy()

    # Convert target variable to binary
    def normalize_binary_target(y):
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = y.astype(str).str.strip()
            vals = list(sorted(y.unique()))
            mapping = {vals[0]: 0, vals[1]: 1}
            return y.map(mapping).astype(int)
        return y.astype(int)

    y_train = normalize_binary_target(y_train)

    # 2. Build preprocessing pipelines
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

    # 3. Define models
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

    # 4. Train models
    models = {
        "Logistic Regression": clone(logistic_best),
        "Random Forest": clone(rf_best),
        "CatBoost": clone(catboost_best)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

# Load models (calls the function above)
models = train_and_get_models()

# ==========================================
# Build Web UI
# ==========================================
st.markdown("---")
st.subheader("📝 Input Clinical Features")

col1, col2 = st.columns(2)

with col1:
    growth_pattern_input = st.selectbox("Growth Pattern", options=["Endoluminal", "Exophytic", "Mixed"])
    ulcer_input = st.selectbox("Ulcer", options=["No", "Yes"])
    length = st.number_input("Length (cm)", min_value=0.0, value=5.0, step=0.1)

with col2:
    tg_hdl = st.number_input("TG/HDL", min_value=0.0, value=1.0, step=0.1)
    sii = st.number_input("SII", min_value=0.0, value=500.0, step=10.0)
    lff = st.number_input("LFF", min_value=0.0, value=10.0, step=1.0)
    vfa = st.number_input("VFA (cm²)", min_value=0.0, value=100.0, step=1.0)

# Variable mapping
gp_mapping = {"Endoluminal": 1, "Exophytic": 2, "Mixed": 3}
ulcer_mapping = {"No": 0, "Yes": 1}

# ==========================================
# Prediction Logic
# ==========================================
if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    # Build input DataFrame (force correct types)
    input_data = pd.DataFrame({
        "Growth pattern": [int(gp_mapping[growth_pattern_input])],
        "Ulcer": [int(ulcer_mapping[ulcer_input])],
        "Length": [float(length)],
        "TG/HDL": [float(tg_hdl)],
        "SII": [float(sii)],
        "LFF": [float(lff)],
        "VFA": [float(vfa)]
    })
    
    # Soft Voting Prediction
    probas = []
    
    for name, model in models.items():
        proba = model.predict_proba(input_data)[:, 1][0]
        probas.append(proba)
        
    # Calculate weighted average probability (weights are all 1)
    final_proba = np.average(probas, weights=[1, 1, 1])
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Display results
    if final_proba >= 0.5:
        st.error(f"**High Risk**")
    else:
        st.success(f"**Low Risk**")
        
    st.metric(label="Probability of High Risk", value=f"{final_proba * 100:.2f}%")
    st.progress(float(final_proba))
    
    # Show individual base model prediction probabilities
    with st.expander("View prediction details of base models"):
        st.write(f"- **Logistic Regression**: {probas[0]*100:.2f}%")
        st.write(f"- **Random Forest**: {probas[1]*100:.2f}%")
        st.write(f"- **CatBoost**: {probas[2]*100:.2f}%")
