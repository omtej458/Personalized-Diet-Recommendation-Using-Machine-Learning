import streamlit as st
import joblib
import pandas as pd
import os, sys

st.title("Model Trained Diet Recommendation System") #This is a data driven model

# sys.path.append("../src")

model = joblib.load("models/RandomizedSearchCV.pkl")

st.subheader("Enter your health details here:")

age = st.number_input("Age", 1, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
height_cm = st.number_input("Height (cm) ", 50, 250, 170)
weight_kg = st.number_input("Weight (kg)", 10, 200, 70)
bmi = weight_kg / (height_cm / 100) ** 2

activity_level = st.selectbox("Activity_level", ["Low", "Moderate", "High"])
goal = st.selectbox("Goal", ['Lose Weight', 'Gain Muscle', 'Maintain', 'Improve Energy'])
medical_condition = st.selectbox("Medical Condition", ["Unknown", "Diabetes", "Thyroid", "Hypertension"])
diet_preference = st.selectbox("Diet Preference",[ 'Veg', 'Non-Veg', 'Vegan', 'Pescatarian'])
water_intake_liters = st.number_input("Water Intake (Liters)", 0.0, 5.0, 2.0)
sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)
hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 18.0, 13.0)
sugar_level = st.selectbox("Sugar level",[ 'Low', 'Normal', 'Slightly High', 'High'])
cholesterol_level = st.selectbox("Cholesterol Level",[ 'Low', 'Borderline', 'High'])

if st.button("Get Diet Recommendation"):
    input_df = pd.DataFrame([
        {
            "age" : age,
            "gender": gender,
            "height_cm" : height_cm,
            "weight_kg" : weight_kg,
            "bmi" : bmi,
            "activity_level" : activity_level,
            "goal" : goal,
            "medical_condition": medical_condition,
            "diet_preference" : diet_preference,
            "water_intake_liters" : water_intake_liters,
            "sleep_hours" : sleep_hours,
            "hemoglobin" : hemoglobin,
            "sugar_level" : sugar_level,
            "cholesterol_level" : cholesterol_level
        }
    ])

    prediction = model.predict(input_df)[0]
    st.success(f"Recommended Diet: **{prediction}**")
