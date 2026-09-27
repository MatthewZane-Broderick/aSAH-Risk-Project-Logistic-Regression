### here I'll load in and collect the feature values for the model

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="aSAH Outcome Risk Prototype",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --accent: #146b6f;
        --accent-dark: #0e4f53;
        --surface: rgba(20, 107, 111, 0.07);
        --border: rgba(127, 127, 127, 0.22);
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .intro {
        color: rgba(127, 127, 127, 0.95);
        font-size: 1rem;
        max-width: 68ch;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 1.35rem 1.5rem;
        margin-top: 1.25rem;
    }
    .result-label {
        color: rgba(127, 127, 127, 0.95);
        font-size: 0.84rem;
        font-weight: 650;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .result-value {
        color: var(--accent);
        font-size: clamp(2rem, 7vw, 3.5rem);
        font-weight: 750;
        line-height: 1.05;
        margin: 0.25rem 0 0.6rem;
        font-variant-numeric: tabular-nums;
    }
    .result-note {
        color: rgba(127, 127, 127, 0.95);
        font-size: 0.9rem;
        margin: 0;
    }
    div[data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 1.25rem;
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 44px;
        border-radius: 0.5rem;
        background: var(--accent);
        border-color: var(--accent);
        color: white;
        font-weight: 650;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sah_logistic_pipeline.joblib"

FEATURES = [
    "age",
    "wfns_grade",
    "fisher_grade",
    "sbp_admission",
    "tmt_mean",
    "nlr",
    "albumin",
    "aneurysm_size_mm",
]


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def poor_outcome_class_index(pipeline) -> int:
    model = pipeline.named_steps.get("model")
    classes = getattr(model, "classes_", None)

    if classes is None:
        classes = getattr(pipeline, "classes_", None)

    if classes is None or 1 not in classes:
        raise ValueError("The fitted classifier does not contain outcome class 1.")

    return int(np.flatnonzero(np.asarray(classes) == 1)[0])


st.markdown('<p class="eyebrow">Research prototype</p>', unsafe_allow_html=True)
st.title("aSAH 6-Month Outcome Estimator")
st.markdown(
    '<p class="intro">Enter the eight admission variables used by the fitted '
    'logistic-regression pipeline. The app returns the model-estimated probability '
    'of poor six-month outcome.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected: "
        f"`{MODEL_PATH}`\n\n"
        "Place `app.py` inside the project’s `app/` folder and keep the saved "
        "pipeline inside `models/`."
    )
    st.stop()

try:
    loaded_clf = load_model(MODEL_PATH)
except Exception as exc:
    st.error(
        "The model could not be loaded. Confirm that the app uses compatible "
        "Python, scikit-learn, pandas, NumPy, and joblib versions."
    )
    st.exception(exc)
    st.stop()

with st.form("patient_inputs", clear_on_submit=False):
    st.subheader("Admission variables")

    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Age (years)", min_value=0, max_value=120, value=60, step=1
        )
        wfns_grade = st.selectbox("WFNS grade", options=[1, 2, 3, 4, 5], index=2)
        fisher_grade = st.selectbox(
            "Fisher grade", options=[1, 2, 3, 4], index=3
        )
        sbp_admission = st.number_input(
            "Admission systolic BP (mmHg)",
            min_value=40.0,
            max_value=350.0,
            value=160.0,
            step=1.0,
        )

    with right:
        tmt_mean = st.number_input(
            "Mean temporal muscle thickness (mm)",
            min_value=0.0,
            max_value=50.0,
            value=5.5,
            step=0.1,
            format="%.1f",
        )
        nlr = st.number_input(
            "Neutrophil-to-lymphocyte ratio",
            min_value=0.0,
            max_value=200.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )
        albumin = st.number_input(
            "Albumin (use the model-training unit)",
            min_value=0.0,
            max_value=100.0,
            value=3.5,
            step=0.1,
            format="%.1f",
            help="Enter albumin in exactly the same unit used in the training CSV.",
        )
        aneurysm_size_mm = st.number_input(
            "Aneurysm size (mm)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.1,
            format="%.1f",
        )

    submitted = st.form_submit_button(
        "Estimate probability", use_container_width=True, type="primary"
    )

if submitted:
    patient = pd.DataFrame(
        [
            {
                "age": age,
                "wfns_grade": wfns_grade,
                "fisher_grade": fisher_grade,
                "sbp_admission": sbp_admission,
                "tmt_mean": tmt_mean,
                "nlr": nlr,
                "albumin": albumin,
                "aneurysm_size_mm": aneurysm_size_mm,
            }
        ],
        columns=FEATURES,
    )

    try:
        class_index = poor_outcome_class_index(loaded_clf)
        probability = float(loaded_clf.predict_proba(patient)[0, class_index])
    except Exception as exc:
        st.error(
            "Prediction failed. Check that the input feature names match those "
            "used when the pipeline was fitted."
        )
        st.exception(exc)
    else:
        st.markdown(
            f"""
            <section class="result-card" aria-live="polite">
                <div class="result-label">Estimated probability of poor outcome</div>
                <div class="result-value">{probability:.1%}</div>
                <p class="result-note">
                    Model output for this entered record; no clinical threshold or
                    treatment recommendation has been applied.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Model and safety notes"):
    st.markdown(
        """
        - This is a research and software demonstration, not a medical device.
        - It has not been established here as externally validated or fit for clinical use.
        - Do not use the output for diagnosis, prognosis, triage, or treatment decisions.
        - Inputs must use the same definitions and units as the training dataset.
        - The displayed value is a model estimate, not a certainty.
        """
    )