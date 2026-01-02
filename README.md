<p align="center">
  <img src="images/banner.png" alt="Personalized Diet Recommendation" width="100%" />
</p>

# 🥗 Personalized Diet Recommendation Using Machine Learning  
*AI-driven nutrition & diet recommendations tailored to your goals and needs*

---

## 🚀 Overview

Welcome to the **Personalized Diet Recommendation System**, an intelligent application that uses machine learning to suggest customized diet plans based on user health data, preferences, and goals. This project aims to empower users with personalized nutrition guidance — whether you're managing weight, optimizing fitness, or improving overall health.

---

## 🧠 Core Features

✨ **Personalized Recommendations**  
Dynamically generates diet suggestions tailored to user profiles including age, weight, height, activity, and dietary preferences.

📊 **Machine Learning Models**  
Utilizes trained models to predict ideal nutrient intake and suggest diet plans based on learned patterns from real data.

📅 **User Inputs & Custom Goals**  
Users can enter health parameters and select dietary goals like *weight loss*, *gain*, or *maintenance*.

📈 **Insightful Outputs**  
Provides clear, actionable meal suggestions and nutritional breakdowns.

---

## 🛠️ Architecture & Workflow

```plaintext
User Input ──▶ Data Preprocessing ──▶ ML Model Prediction  
     │                                           │
     ▼                                           ▼
Preference Check ──▶ Diet Generation ──▶ Results + Visual Output
```
## 🧱 Project Structure

```
Personalized-Diet-Recommendation-Using-Machine-Learning/
    ├── .github/
    │   ├── workflows/
    │   │   ├── main_personalizeddietrecommendation.yml
    ├── app/
    │   ├── data/
    │   │   ├── processed/
    │   │   │   ├── test.csv
    │   │   │   └── train.csv
    │   │   ├── generated_diet_data.csv
    │   ├── images/
    │   │   ├── attribute_histogram_plots.png
    │   │   ├── Numeric Correlation.png
    │   │   ├── Outliers fixed plot.png
    ├── data/
    │   ├── processed/
    │   │   ├── test.csv
    │   │   ├── train.csv
    │   ├── generated_diet_data.csv
    ├── images/
    │   ├── attribute_histogram_plots.png
    │   ├── Numeric Correlation.png
    │   ├── Outliers fixed plot.png
    ├── models/
    │   ├── RandomForestClassifier.pkl
    │   ├── RandomizedSearchCV.pkl
    ├── notebooks/
    │   ├── 1.data_generation.ipynb
    │   ├── 2.data_exploration.ipynb
    │   ├── data_preprocessing.ipynb
    │   ├── model_traval.ipynb
    ├── src/
    │   ├── __pycache__/
    │   │   ├── data_cleaning.cpython-311.pyc
    │   │   ├── data_generation.cpython-311.pyc
    │   │   └── data_preprocessing.cpython-311.pyc
    │   ├── data_cleaning.py
    │   ├── data_generation.py
    │   └── data_preprocessing.py
    ├── app.py
    ├── README.md
    └── requirements.txt
```

## 🔧 Installation & Setup

Follow these steps to get started locally:

```bash
git clone https://github.com/omtej458/Personalized-Diet-Recommendation-Using-Machine-Learning.git
cd Personalized-Diet-Recommendation-Using-Machine-Learning
pip install -r requirements.txt
```
📌 Tip: Use a virtual environment for isolation:
```bash
python3 -m venv venv
source venv/bin/activate
```



## 🧪 Running the Application

Start the app using:
```bash
streamlit run app.py
```

## 🔗 Once the command is run automatically browser will open:
```
example:
http://localhost:5000
```
 ## 📊 Examples (Visuals)
🍎 Sample Meal Recommendation

📈 Nutrient Breakdown

## 🧪 Model Training Overview
We trained models using nutritional datasets with labels tied to user goals. Techniques used:
Regression models to determine macro nutrient targets
Classification to assign diet types
Feature scaling and normalization

Models are stored under models/ and loaded at runtime for predictions.

## 🧩 Future Improvements

🌟 Add user accounts & authentication <br>
🌟 Enhance the UI with interactive dashboards <br>
🌟 Extend to fitness recommendations (workouts + diet) <br>
🌟 Deploy on cloud (Heroku / AWS / GCP) <br>

<p align="center"> Made with ❤️ • <a href="https://github.com/omtej458/Personalized-Diet-Recommendation-Using-Machine-Learning">Personalized Diet Recommendation Repo</a> </p>
