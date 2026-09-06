import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Loads and caches the model and metadata series."""
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("embarked.pkl", "rb") as f:
        embarked_series = pickle.load(f)
    with open("sex.pkl", "rb") as f:
        sex_series = pickle.load(f)
    return model, embarked_series, sex_series

try:
    model, embarked_series, sex_series = load_resources()
except FileNotFoundError as e:
    st.error(f"Missing required pickle file: {e}. Ensure all .pkl files are in the working directory.")
    st.stop()

# --- User Interface ---
st.title("🚢 Titanic Survival Prediction")
st.write("Adjust the passenger details below to predict survival probability.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=2)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
    sex = st.selectbox("Sex", options=["female", "male"])

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q", "1"])

st.divider()

# --- Feature Mapping ---
# Exact feature array expected by model fit: 
# ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_female', 'Sex_male', 'Embarked_1', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
input_data = {
    'Pclass': int(pclass),
    'Age': float(age),
    'SibSp': int(sibsp),
    'Parch': int(parch),
    'Sex_female': 1 if sex == "female" else 0,
    'Sex_male': 1 if sex == "male" else 0,
    'Embarked_1': 1 if embarked == "1" else 0,
    'Embarked_C': 1 if embarked == "C" else 0,
    'Embarked_Q': 1 if embarked == "Q" else 0,
    'Embarked_S': 1 if embarked == "S" else 0,
}

features_df = pd.DataFrame([input_data])

# --- Prediction Action ---
if st.button("Predict Survival", type="primary", use_container_width=True):
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    survival_prob = probabilities[1] * 100

    if prediction == 1:
        st.success("Result: **Survived** 🎉")
        st.metric(label="Estimated Survival Probability", value=f"{survival_prob:.1f}%")
    else:
        st.error("Result: **Did Not Survive** ❌")
        st.metric(label="Estimated Survival Probability", value=f"{survival_prob:.1f}%")

    with st.expander("View Input Vector"):
        st.dataframe(features_df)