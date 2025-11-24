def clean_dataset(df):
    # 1. Fix outliers
    def fix_outliers(df, col):
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        med = df[col].median()
        df.loc[(df[col] < lower) | (df[col] > upper), col] = med
        return df
    
    for col in ["age", "height_cm", "weight_kg", "bmi", "hemoglobin"]:
        df = fix_outliers(df, col)

    # 2. Fix NaN in medical_condition
    df["medical_condition"] = df["medical_condition"].fillna("Unknown")

    return df
