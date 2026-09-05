import streamlit as st
import pickle
import pandas as pd

# 1. Load the model AND both encoders (done once when the app starts)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("city_encoder.pkl", "rb") as f:
    city_encoder = pickle.load(f)

with open("membership_encoder.pkl", "rb") as f:
    membership_encoder = pickle.load(f)

# 2. Page title
st.title("Customer Purchase Predictor")
st.write("Fill in the details and click Predict.")

# 3. Numeric inputs -> typed directly as numbers
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)

# 4. Text inputs -> dropdowns filled from each encoder's known categories.
#    Using .classes_ means the user can ONLY pick values the model has seen.
city = st.selectbox("City", city_encoder.classes_)
membership = st.selectbox("Membership", membership_encoder.classes_)

# 5. Predict button
if st.button("Predict"):
    # 5a. Translate the chosen words into numbers using the SAME saved encoders
    city_num = city_encoder.transform([city])[0]
    membership_num = membership_encoder.transform([membership])[0]

    # 5b. Build one row in the SAME column order used during training
    row = pd.DataFrame(
        [[age, income, city_num, membership_num]],
        columns=["age", "income", "city_encoded", "membership_encoded"]
    )

    # 5c. Predict, and also show the probability
    result = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    if result == 1:
        st.success(f"Likely to purchase  (probability {proba:.0%})")
    else:
        st.info(f"Unlikely to purchase  (probability {proba:.0%})")
