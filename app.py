import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict whether the customer will churn.")

# User Inputs
credit_score = st.number_input("Credit Score", 300, 900, 600)
country = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
products_number = st.slider("Number of Products", 1, 4, 1)
credit_card = st.selectbox("Has Credit Card", [0,1])
active_member = st.selectbox("Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Encoding categorical variables
country_map = {"France":0,"Spain":1,"Germany":2}
gender_map = {"Male":1,"Female":0}

country = country_map[country]
gender = gender_map[gender]

# Create input array
input_data = np.array([[credit_score,country,gender,age,tenure,balance,
                        products_number,credit_card,active_member,estimated_salary]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer will NOT churn")