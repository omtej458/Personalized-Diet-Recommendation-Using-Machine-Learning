import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("../app/data", exist_ok = True)

n_samples = 2500

age = np.clip(np.random.normal(
    loc = 32,
    scale = 10,
    size = n_samples).astype(int),18,65
)
height_base = np.random.normal(
    loc = 167,
    scale = 8,
    size = n_samples
)
gender = np.random.choice(
    ['Male', 'Female'],
    size = n_samples,
    p = [0.49, 0.51]
)
height = height_base + np.where(gender == 'Male', 6, -2)
height = np.clip(height, 140, 200).round(1)

bmi = np.clip(np.random.normal(
    loc = 24.5,
    scale = 3.8,
    size = n_samples),16.0, 40.0).round(1)

weight = (bmi * (height/100) ** 2).round(1)

hb_mean = np.where(gender == 'Male', 14.2, 12.8)
hemoglobin = np.clip(np.random.normal(
    loc = hb_mean,
    scale = 1.2,
    size = n_samples), 8.0, 18.0).round(1)

water_intake = np.clip(np.random.normal(
    loc = 2.2,
    scale = 0.6,
    size = n_samples), 0.5, 5.0).round(2)

sleep_hours = np.clip(np.random.normal(
    loc = 7.0,
    scale = 1.2,
    size = n_samples), 3.5, 10.0).round(1)

activity_probs = []
for a in age:
    if a < 25:
        activity_probs.append([0.15, 0.35, 0.35, 0.15])
    elif a < 40:
        activity_probs.append([0.25, 0.35, 0.30, 0.10])
    else:
        activity_probs.append([0.35, 0.40, 0.20, 0.05])
activity_level = [np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], p=p) for p in activity_probs]

goal = np.random.choice(['Lose Weight', 'Gain Muscle', 'Maintain', 'Improve Energy'],
                        size=n_samples, p=[0.42, 0.18, 0.30, 0.10])

diet_preference = np.random.choice(['Veg', 'Non-Veg', 'Vegan', 'Pescatarian'],
                                   size=n_samples, p=[0.45, 0.40, 0.08, 0.07])

medical_condition = []
for a, b in zip(age, bmi):
    base = np.random.random()
    if base < 0.02:
        medical_condition.append('Thyroid')
    elif base < (0.02 + (0.06 if a > 50 else 0.02)):
        medical_condition.append('Hypertension')
    elif base < (0.02 + 0.06 + (0.06 if b > 27 else 0.02)):
        medical_condition.append('Diabetes')
    else:
        medical_condition.append('None')

sugar_level = []
for cond, b in zip(medical_condition, bmi):
    if cond == 'Diabetes':
        sugar_level.append(np.random.choice(['Slightly High', 'High'], p=[0.4, 0.6]))
    else:
        if b < 25:
            p_normal = 0.75
            p_slight = 0.15
            p_high = 0.05
        else:
            p_normal = 0.60
            p_slight = 0.25
            p_high = 0.10
        sugar_level.append(np.random.choice(['Low', 'Normal', 'Slightly High', 'High'],
                                            p=[0.05, p_normal, p_slight, p_high]))

cholesterol_level = []
for a, b in zip(age, bmi):
    if a > 55 or b > 30:
        cholesterol_level.append(np.random.choice(['Normal', 'Borderline', 'High'], p=[0.25, 0.45, 0.30]))
    elif a > 40 or b > 27:
        cholesterol_level.append(np.random.choice(['Normal', 'Borderline', 'High'], p=[0.45, 0.40, 0.15]))
    else:
        cholesterol_level.append(np.random.choice(['Normal', 'Borderline', 'High'], p=[0.7, 0.25, 0.05]))

recommended_diet = []
for hb, g, bmi_val, sugar, chol, cond, act in zip(hemoglobin, goal, bmi, sugar_level, cholesterol_level, medical_condition, activity_level):
    if hb < 11.0:
        recommended_diet.append('Iron-Rich Diet')
        continue
    if cond == 'Diabetes':
        if sugar in ['High', 'Slightly High'] or bmi_val > 27:
            recommended_diet.append('Low-Carb Diet')
            continue
        else:
            recommended_diet.append('Balanced Diet')
            continue
    if chol == 'High':
        recommended_diet.append('Heart-Healthy Diet')
        continue
    if g == 'Gain Muscle':
        if act in ['Active', 'Moderate'] and bmi_val < 24.5:
            recommended_diet.append('High-Protein Diet')
        else:
            recommended_diet.append('Balanced Diet')
        continue
    if g == 'Lose Weight':
        if bmi_val > 27 or sugar in ['Slightly High', 'High']:
            recommended_diet.append('Low-Carb Diet')
        else:
            recommended_diet.append('Balanced Diet')
        continue
    recommended_diet.append('Balanced Diet')

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'height_cm': height,
    'weight_kg': weight,
    'bmi': bmi,
    'activity_level': activity_level,
    'goal': goal,
    'medical_condition': medical_condition,
    'diet_preference': diet_preference,
    'water_intake_liters': water_intake,
    'sleep_hours': sleep_hours,
    'hemoglobin': hemoglobin,
    'sugar_level': sugar_level,
    'cholesterol_level': cholesterol_level,
    'recommended_diet': recommended_diet #key
})

out_path = "../app/data/generated_diet_data.csv"
df.to_csv(out_path, index=False)


print("✅ Saved to:", out_path)
print("\nClass distribution (recommended_diet):\n")

print(df['recommended_diet'].value_counts(normalize=False))