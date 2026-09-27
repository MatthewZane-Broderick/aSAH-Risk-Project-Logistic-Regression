import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


num_features =[
    'age',
    'wfns_grade',
    'fisher_grade',
    'sbp_admission',
    'tmt_mean',
    'nlr',
    'albumin',
    'aneurysm_size_mm'
    ]

TARGET = 'poor_outcome_6m'
def load_data(path: str):
    df = pd.read_csv(path)

    missing_values = df[num_features].isna().sum()
    if missing_values.any():
        print("Missing values found in the following numerical features:")
        print(missing_values[missing_values > 0])
    return df


def build_preprocessor(scale_for_logistic: bool = True):
    steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]

    if scale_for_logistic:
        steps.append(('scaler', StandardScaler()))

    num_pipe = Pipeline(steps)

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features)
    ], remainder='drop')

    return preprocessor


def load_and_split_data(path:str, target_col: str = TARGET,
                        test_size: float = 0.2, random_state: int = 42):
    df = load_data(path)

    X = df[num_features]
    Y = df[target_col]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, Y_train, Y_test