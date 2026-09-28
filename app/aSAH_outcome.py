from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd


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


def load_model():
    return joblib.load(MODEL_PATH)


def poor_outcome_class_index(pipeline):
    model = pipeline.named_steps.get("model")
    classes = getattr(model, "classes_", None)

    if classes is None:
        classes = getattr(pipeline, "classes_", None)

    if classes is None or 1 not in classes:
        raise ValueError("Outcome class 1 not found.")

    return int(np.flatnonzero(np.asarray(classes) == 1)[0])


clf = load_model()
class_index = poor_outcome_class_index(clf)


def predict():
    try:
        patient = pd.DataFrame([{
            "age": float(entries["age"].get()),
            "wfns_grade": float(entries["wfns_grade"].get()),
            "fisher_grade": float(entries["fisher_grade"].get()),
            "sbp_admission": float(entries["sbp_admission"].get()),
            "tmt_mean": float(entries["tmt_mean"].get()),
            "nlr": float(entries["nlr"].get()),
            "albumin": float(entries["albumin"].get()),
            "aneurysm_size_mm": float(entries["aneurysm_size_mm"].get()),
        }], columns=FEATURES)

        probability = clf.predict_proba(patient)[0, class_index]
        result_var.set(f"Estimated probability of poor 6-month outcome: {probability:.1%}")

    except Exception as exc:
        messagebox.showerror("Prediction error", str(exc))


root = tk.Tk()
root.title("aSAH Outcome Estimator")
root.geometry("560x520")

main = ttk.Frame(root, padding=20)
main.pack(fill="both", expand=True)

title = ttk.Label(main, text="aSAH 6-Month Outcome Estimator", font=("Segoe UI", 16, "bold"))
title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

labels = {
    "age": "Age (years)",
    "wfns_grade": "WFNS grade",
    "fisher_grade": "Fisher grade",
    "sbp_admission": "Admission systolic BP",
    "tmt_mean": "Mean TMT",
    "nlr": "NLR",
    "albumin": "Albumin",
    "aneurysm_size_mm": "Aneurysm size (mm)",
}

defaults = {
    "age": "60",
    "wfns_grade": "3",
    "fisher_grade": "4",
    "sbp_admission": "160",
    "tmt_mean": "5.5",
    "nlr": "6.0",
    "albumin": "3.5",
    "aneurysm_size_mm": "8.0",
}

entries = {}

for i, feature in enumerate(FEATURES, start=1):
    ttk.Label(main, text=labels[feature]).grid(row=i, column=0, sticky="w", pady=6, padx=(0, 12))
    entry = ttk.Entry(main, width=18)
    entry.insert(0, defaults[feature])
    entry.grid(row=i, column=1, sticky="ew", pady=6)
    entries[feature] = entry

main.columnconfigure(1, weight=1)

ttk.Button(main, text="Estimate probability", command=predict).grid(
    row=len(FEATURES) + 1,
    column=0,
    columnspan=2,
    sticky="ew",
    pady=(18, 12),
)

result_var = tk.StringVar(value="Enter values and click Estimate probability.")
ttk.Label(
    main,
    textvariable=result_var,
    wraplength=500,
    foreground="darkgreen",
).grid(
    row=len(FEATURES) + 2,
    column=0,
    columnspan=2,
    sticky="w",
)

ttk.Label(
    main,
    text="Research prototype only. Not for clinical decision-making.",
    foreground="firebrick",
    wraplength=500,
).grid(
    row=len(FEATURES) + 3,
    column=0,
    columnspan=2,
    sticky="w",
    pady=(14, 0),
)

root.mainloop()