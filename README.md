# aSAH‑Risk Project
an end‑to‑end, explainable ML pipeline for predicting a clinically meaningful aSAH outcome (e.g., DCI or poor functional outcome) plus a Streamlit CDSS demo.


Title:
aSAH‑Risk: An explainable machine‑learning clinical decision support tool for predicting delayed cerebral ischaemia (DCI) after aneurysmal subarachnoid haemorrhage

Can an interpretable ML model, using admission clinical variables and routine labs, accurately predict which aSAH patients will develop DCI, and can this be wrapped in a simple CDSS prototype to support early risk stratification?

DCI is a leading cause of secondary injury and poor outcome after aSAH.

Many recent ML papers already model DCI using clinical + inflammatory markers, so you can benchmark against published performance (AUROC ~0.80–0.90).


Cohort definition:

Include admissions with ICD‑9/10 codes for non‑traumatic subarachnoid haemorrhage (e.g., ICD‑9: 430; ICD‑10: I60.x).
Exclude traumatic SAH and non‑aneurysmal causes if possible

Outcome:

Primary: DCI (if documented in notes or via vasospasm/intervention codes).

Secondary: in‑hospital mortality, poor discharge disposition, or prolonged ICU stay.

Features (examples):

Demographics: age, sex.

Admission severity: GCS (from charted neuro checks), mechanical ventilation status.

Imaging grade proxy: if not available, use surrogates (e.g., EVD placement, early neurosurgical intervention).

Labs (first 24h): WBC, CRP (if available), neutrophil/lymphocyte counts → compute NLR, platelet count → PLR, creatinine, sodium.

Comorbidities: hypertension, diabetes, smoking status (if available).

daily commit 2: 
because of acess to Suwa dataset is available instead of predicting DCI, instead a poor functional outcomes will be predicted