import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_generation import df

from data_cleaning import clean_dataset
# df = pd.read_csv("../data/generated_diet_data.csv")
df = clean_dataset(df)


# df = pd.read_csv("..\data\generated_diet_data.csv")

def train_test_split(df, target ="recommended_diet", test_size = 0.2, random_state = 42):

    X = df.drop(columns = [target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=random_state,
                                                        test_size=test_size,
                                                        stratify = y)

    train_df = X_train.copy()
    train_df[target] = y_train

    test_df = X_test.copy()
    test_df[target] = y_test

    return train_df, test_df

def build_preprocessor(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if "recommended_diet" in cat_cols:
        cat_cols.remove("recommended_diet")

    #numerical preprocessing
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    #category preprocessing
    categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    #combine both 
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols