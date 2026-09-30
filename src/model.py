from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_logistic_pipeline(max_iter: int = 1000):
    preprocessor = build_preprocessor(scale_for_logistic=True)

    log_reg = LogisticRegression(max_iter=max_iter)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", log_reg),
    ])