import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Nutri_Track Dashboard")

st.header("Health Worker Data Entry Form")

with st.form("child_form"):
    age = st.number_input("Age (months)", min_value=1)
    weight = st.number_input("Weight (kg)", min_value=1.0)
    height = st.number_input("Height (cm)", min_value=30.0)
    area = st.text_input("Village / Area")

    submit = st.form_submit_button("Submit & Predict")

if submit:
    prediction = model.predict([[age, weight, height]])
    st.success(f"Predicted Risk Level: {prediction[0]}")
