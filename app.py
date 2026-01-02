# import streamlit as st
# import joblib
# import pandas as pd
# import os, sys

# st.title("Model Trained Diet Recommendation System") #This is a data driven model

# # sys.path.append("../src")

# model = joblib.load("models/RandomizedSearchCV.pkl")

# st.subheader("Enter your health details here:")

# age = st.number_input("Age", 1, 100, 25)
# gender = st.selectbox("Gender", ["Male", "Female"])
# height_cm = st.number_input("Height (cm) ", 50, 250, 170)
# weight_kg = st.number_input("Weight (kg)", 10, 200, 70)
# bmi = weight_kg / (height_cm / 100) ** 2

# activity_level = st.selectbox("Activity_level", ["Low", "Moderate", "High"])
# goal = st.selectbox("Goal", ['Lose Weight', 'Gain Muscle', 'Maintain', 'Improve Energy'])
# medical_condition = st.selectbox("Medical Condition", ["Unknown", "Diabetes", "Thyroid", "Hypertension"])
# diet_preference = st.selectbox("Diet Preference",[ 'Veg', 'Non-Veg', 'Vegan', 'Pescatarian'])
# water_intake_liters = st.number_input("Water Intake (Liters)", 0.0, 5.0, 2.0)
# sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)
# hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 18.0, 13.0)
# sugar_level = st.selectbox("Sugar level",[ 'Low', 'Normal', 'Slightly High', 'High'])
# cholesterol_level = st.selectbox("Cholesterol Level",[ 'Low', 'Borderline', 'High'])

# if st.button("Get Diet Recommendation"):
#     input_df = pd.DataFrame([
#         {
#             "age" : age,
#             "gender": gender,
#             "height_cm" : height_cm,
#             "weight_kg" : weight_kg,
#             "bmi" : bmi,
#             "activity_level" : activity_level,
#             "goal" : goal,
#             "medical_condition": medical_condition,
#             "diet_preference" : diet_preference,
#             "water_intake_liters" : water_intake_liters,
#             "sleep_hours" : sleep_hours,
#             "hemoglobin" : hemoglobin,
#             "sugar_level" : sugar_level,
#             "cholesterol_level" : cholesterol_level
#         }
#     ])

#     prediction = model.predict(input_df)[0]
#     st.success(f"Recommended Diet: **{prediction}**")

import streamlit as st
import joblib
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time,os

# ============================================
# Cache resources for faster loading
# ============================================
@st.cache_resource
def load_ml_model():
    return joblib.load("models/RandomizedSearchCV.pkl")

# @st.cache_resource
# def initialize_llm():
#     load_dotenv()
#     return AzureChatOpenAI(
#         azure_deployment="gpt-5-chat",
#         api_version="2024-12-01-preview",
#         temperature=0.3
#     )

@st.cache_resource
def initialize_llm():
    return AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-08-01-preview"
)

model = load_ml_model()
llm = initialize_llm()

# Prompt template (same as before)
Prompt = PromptTemplate(
    template="""
Act as a highly experienced professional nutritionist and medical doctor. You will receive the following user health and lifestyle details:
{age}, {activity_level}, {gender}, {height_cm}, {weight_kg}, {goal}, {medical_condition}, {diet_preference}, {water_intake_liters}, {sleep_hours}, {cholesterol_level}, {sugar_level}, {hemoglobin}, {Country}.
You will also receive a one-word dietary focus called {prediction} (for example: Balanced Diet, Iron-Rich Diet, Low-Cholesterol Diet, Diabetic Diet, etc.).

Using all the information provided, follow these instructions:

Understand the prediction: Interpret the meaning of {prediction} and tailor the plan to support that nutritional focus while also respecting the user's goal and medical conditions.

Create a personalized diet plan in a tabular format with the following columns:

Meal Time

Food Items (Specific & culturally appropriate to the user's Country)

Portion Size

Nutritional Purpose / Health Benefit

Include: Breakfast, Mid-Morning Snack, Lunch, Evening Snack, Dinner, and (if suitable) Bedtime Option.

Explain & Justify the Diet: After the table, provide a section titled
"Why These Foods Were Selected (Medical & Nutritional Reasoning)"
Give specific, evidence-based reasons linking foods to:

the user's health metrics (cholesterol, sugar level, hemoglobin, etc.)

medical conditions

fitness goal (weight loss, muscle gain, maintenance, etc.)

dietary preference and lifestyle

the meaning of {prediction}

Ensure the recommendations are:

Safe and medically appropriate

Realistic and practical

Culturally relevant

Avoid foods restricted by the user's diet preference or medical condition

Where necessary, include gentle safety disclaimers such as consulting a doctor before major diet changes, especially for serious health conditions.

Output the final response in the following structure:

Output Format

Personalized Recommendation of diet based on {prediction}

Personalized Diet Plan weekly-day to day (Table)

Why These Foods Were Selected (Medical & Nutritional Justification)

Helpful Lifestyle Suggestions (optional, if relevant)
""",
    input_variables=['age', 'activity_level', 'gender', 'height_cm', 'weight_kg', 'goal', 'medical_condition', 'diet_preference', 'water_intake_liters', 'sleep_hours', 'cholesterol_level', 'sugar_level', 'hemoglobin', 'Country', 'prediction']
)

parser = StrOutputParser()
chain = Prompt | llm | parser

# ============================================
# UI
# ============================================
st.title("🥗 AI-Powered Diet Recommendation System")

st.subheader("Enter your health details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height_cm = st.number_input("Height (cm)", 50, 250, 170)
    weight_kg = st.number_input("Weight (kg)", 10, 200, 70)
    bmi = weight_kg / (height_cm / 100) ** 2
    st.info(f"📊 Your BMI: {bmi:.1f}")
    
    activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
    goal = st.selectbox("Goal", ['Lose Weight', 'Gain Muscle', 'Maintain', 'Improve Energy'])

with col2:
    medical_condition = st.selectbox("Medical Condition", ["Unknown", "Diabetes", "Thyroid", "Hypertension"])
    diet_preference = st.selectbox("Diet Preference", ['Veg', 'Non-Veg', 'Vegan', 'Pescatarian'])
    water_intake_liters = st.number_input("Water Intake (Liters)", 0.0, 5.0, 2.0)
    sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 18.0, 13.0)
    sugar_level = st.selectbox("Sugar Level", ['Low', 'Normal', 'Slightly High', 'High'])
    cholesterol_level = st.selectbox("Cholesterol Level", ['Low', 'Borderline', 'High'])
    Country = st.selectbox("Country", ['Japan', 'Vietnam', 'Thailand', 'Singapore', 'China', 'India', 'Turkey', 'Italy', 'Spain', 'France', 'Mexico', 'USA'])

# ============================================
# Enhanced button with status tracking
# ============================================
if st.button("🎯 Get Diet Recommendation", type="primary", use_container_width=True):
    # Create input dataframe
    input_df = pd.DataFrame([
        {
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "activity_level": activity_level,
            "goal": goal,
            "medical_condition": medical_condition,
            "diet_preference": diet_preference,
            "water_intake_liters": water_intake_liters,
            "sleep_hours": sleep_hours,
            "hemoglobin": hemoglobin,
            "sugar_level": sugar_level,
            "cholesterol_level": cholesterol_level
        }
    ])

    # Enhanced status tracking
    with st.status("🔄 Preparing your personalized diet plan...", expanded=True) as status:
        # Step 1: Analyze health data
        st.write("🔍 **Step 1:** Analyzing your health metrics...")
        prediction = model.predict(input_df)[0]
        time.sleep(0.3)
        st.write(f"✅ Analysis complete! Dietary focus: **{prediction}**")
        
        # Step 2: AI consultation
        st.write("🤖 **Step 2:** Consulting AI nutritionist...")
        time.sleep(0.3)
        st.write("✅ AI nutritionist ready!")
        
        # Step 3: Generating plan
        st.write("📝 **Step 3:** Generating your customized diet plan...")
        time.sleep(0.2)
        
        status.update(label="✨ Your diet plan is ready!", state="complete")
    
    # Display the result
    st.divider()
    st.subheader(f"📋 Your Personalized Diet Plan")
    
    # Show key info in metrics
    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.metric("Dietary Focus", prediction)
    with met_col2:
        st.metric("Goal", goal)
    with met_col3:
        st.metric("BMI", f"{bmi:.1f}")
    
    st.divider()
    
    # Stream the diet plan
    with st.container():
        st.write_stream(
            chain.stream({
                'age': age,
                'activity_level': activity_level,
                'gender': gender,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'goal': goal,
                'medical_condition': medical_condition,
                'diet_preference': diet_preference,
                'water_intake_liters': water_intake_liters,
                'sleep_hours': sleep_hours,
                'cholesterol_level': cholesterol_level,
                'sugar_level': sugar_level,
                'hemoglobin': hemoglobin,
                'Country': Country,
                'prediction': prediction
            })
        )
    
    st.success("✅ Diet plan generated successfully! Feel free to ask questions or request modifications.")
    
    # # Optional: Add download button
    # result_text = "..." # Store the result
    # st.download_button("📥 Download Diet Plan", result_text, file_name="my_diet_plan.txt")
