"""
Diabetes Patient Readmission Prediction
Streamlit app deploying the best model: Optimized Gradient Boosting Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: load artifacts
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load scaler and best GBC model from pickle files."""
    scaler = pickle.load(open("pickle-files/scaler.sav", "rb"))
    model  = pickle.load(open("pickle-files/best_classifier.pkl", "rb"))
    return scaler, model


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors the notebook exactly
# ──────────────────────────────────────────────────────────────────────────────

COLS_NUM = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

COLS_CAT = [
    "race", "gender", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "payer_code",
]

COLS_CAT_NUM = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

TOP_10_SPECIALTIES = [
    "UNK", "InternalMedicine", "Emergency/Trauma",
    "Family/GeneralPractice", "Cardiology", "Surgery-General",
    "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist",
]

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 10, "[20-30)": 20, "[30-40)": 30,
    "[40-50)": 40, "[50-60)": 50, "[60-70)": 60,
    "[70-80)": 70, "[80-90)": 80, "[90-100)": 90,
}

DRUG_OPTIONS  = ["No", "Down", "Steady", "Up"]
RESULT_OPTIONS = ["None", "Norm", ">200", ">300"]
A1C_OPTIONS    = ["None", "Norm", ">7", ">8"]
RACE_OPTIONS   = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "UNK"]
GENDER_OPTIONS = ["Male", "Female"]
CHANGE_OPTIONS = ["No", "Ch"]
MED_OPTIONS    = ["Yes", "No"]
PAYER_OPTIONS  = [
    "UNK", "MC", "HM", "BC", "SP", "CP", "SI", "DM",
    "CH", "PO", "WC", "OT", "OG", "UN", "FR",
]
SPECIALTY_OPTIONS = TOP_10_SPECIALTIES + ["Other"]
ADMISSION_TYPE   = [str(i) for i in range(1, 9)]
DISCHARGE_DISP   = [str(i) for i in list(range(1, 29)) if i not in [11, 13, 14, 19, 20, 21]]
ADMISSION_SOURCE = [str(i) for i in range(1, 26)]
AGE_BUCKETS      = list(AGE_MAP.keys())


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Convert UI inputs into a single-row DataFrame matching training features."""
    row = {}

    # Numerical
    for c in COLS_NUM:
        row[c] = inputs[c]

    # Categorical IDs as strings (for get_dummies)
    for c in COLS_CAT_NUM:
        row[c] = str(inputs[c])

    # Plain categorical
    for c in COLS_CAT:
        row[c] = inputs[c]

    # Medical specialty bucket
    spec = inputs["medical_specialty"]
    row["med_spec"] = spec if spec in TOP_10_SPECIALTIES else "Other"

    # Age group
    row["age_group"] = AGE_MAP[inputs["age"]]

    # Has weight flag
    row["has_weight"] = int(inputs["has_weight"])

    df = pd.DataFrame([row])

    # One-hot encode categorical columns (drop_first=True to match training)
    df_cat = pd.get_dummies(df[COLS_CAT + COLS_CAT_NUM + ["med_spec"]], drop_first=True)

    return df, df_cat


def align_to_training(df_num: pd.DataFrame, df_cat: pd.DataFrame, scaler) -> np.ndarray:
    """
    Reconstruct the full feature vector the scaler was fitted on.
    Missing dummy columns default to 0; extra columns are dropped.
    """
    # Retrieve the feature names the scaler was fit on
    scaler_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

    cols_extra = ["age_group", "has_weight"]
    df_numeric  = df_num[COLS_NUM + cols_extra].reset_index(drop=True)
    df_combined = pd.concat([df_numeric, df_cat.reset_index(drop=True)], axis=1)

    if scaler_features is not None:
        # Reindex to match training columns exactly
        df_combined = df_combined.reindex(columns=scaler_features, fill_value=0)
    
    return df_combined.values


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.title("🏥 Diabetes Patient Readmission Predictor")
st.markdown(
    """
    This tool uses an **Optimized Gradient Boosting Classifier** (best AUC on validation set)
    trained on the UCI Diabetes 130-US Hospitals dataset to estimate the probability that a
    diabetic patient will be **readmitted within 30 days**.

    Fill in the patient details below and click **Predict**.
    """
)

# Check artifacts exist
artifacts_ok = (
    os.path.exists("pickle-files/scaler.sav") and
    os.path.exists("pickle-files/best_classifier.pkl")
)

if not artifacts_ok:
    st.error(
        "⚠️ Model artifacts not found. Make sure `pickle-files/scaler.sav` and "
        "`pickle-files/best_classifier.pkl` are present in the working directory."
    )
    st.stop()

scaler, model = load_artifacts()

# ── Sidebar: About ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Model:** Gradient Boosting Classifier (sklearn)  
        **Tuning:** RandomizedSearchCV (20 iterations, 2-fold CV)  
        **Metric:** ROC-AUC  
        **Test AUC:** ~0.67  

        **Top predictive features:**
        - `number_inpatient` — prior inpatient visits
        - `discharge_disposition_id` — discharge setting
        - `num_medications` — number of medications
        - `num_lab_procedures` — lab work ordered
        - `time_in_hospital` — length of stay
        """
    )
    st.markdown("---")
    st.caption("Source notebook: Readmission_Prediction.ipynb")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("patient_form"):
    st.subheader("Patient Demographics")
    c1, c2, c3 = st.columns(3)
    age    = c1.selectbox("Age group", AGE_BUCKETS, index=6)
    race   = c2.selectbox("Race", RACE_OPTIONS)
    gender = c3.selectbox("Gender", GENDER_OPTIONS)

    st.subheader("Admission Details")
    c4, c5, c6 = st.columns(3)
    admission_type_id       = c4.selectbox("Admission type ID", ADMISSION_TYPE, index=0)
    discharge_disposition_id = c5.selectbox("Discharge disposition ID", DISCHARGE_DISP, index=0)
    admission_source_id     = c6.selectbox("Admission source ID", ADMISSION_SOURCE, index=6)
    medical_specialty       = st.selectbox("Medical specialty", SPECIALTY_OPTIONS)

    st.subheader("Hospital Stay Metrics")
    c7, c8, c9, c10 = st.columns(4)
    time_in_hospital   = c7.number_input("Time in hospital (days)", 1, 14, 3)
    num_lab_procedures = c8.number_input("Lab procedures", 1, 132, 44)
    num_procedures     = c9.number_input("Procedures", 0, 6, 1)
    num_medications    = c10.number_input("Medications", 1, 81, 15)

    c11, c12, c13, c14 = st.columns(4)
    number_outpatient = c11.number_input("Outpatient visits (yr)", 0, 42, 0)
    number_emergency  = c12.number_input("Emergency visits (yr)", 0, 76, 0)
    number_inpatient  = c13.number_input("Inpatient visits (yr)", 0, 21, 0)
    number_diagnoses  = c14.number_input("Number of diagnoses", 1, 16, 7)

    st.subheader("Lab Results")
    c15, c16 = st.columns(2)
    max_glu_serum = c15.selectbox("Max glucose serum", RESULT_OPTIONS)
    A1Cresult     = c16.selectbox("A1C result", A1C_OPTIONS)

    st.subheader("Medications")
    med_cols = st.columns(6)
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    med_inputs = {}
    for i, med in enumerate(med_names):
        med_inputs[med] = med_cols[i % 6].selectbox(med, DRUG_OPTIONS, key=med)

    st.subheader("Other")
    c17, c18, c19, c20 = st.columns(4)
    change      = c17.selectbox("Medication change", CHANGE_OPTIONS)
    diabetesMed = c18.selectbox("On diabetes medication", MED_OPTIONS)
    payer_code  = c19.selectbox("Payer code", PAYER_OPTIONS)
    has_weight  = c20.checkbox("Weight recorded?", value=False)

    threshold = st.slider(
        "Decision threshold (probability ≥ threshold → Readmitted)",
        0.1, 0.9, 0.5, 0.05,
    )

    submitted = st.form_submit_button("🔍 Predict Readmission Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        # Numerical
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        # Demographics
        "age": age,
        "race": race,
        "gender": gender,
        "has_weight": has_weight,
        # Admission
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "medical_specialty": medical_specialty,
        # Lab
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        # Medications
        **med_inputs,
        # Other
        "change": change,
        "diabetesMed": diabetesMed,
        "payer_code": payer_code,
        # Combos not in UI — default "No"
        "glyburide-metformin": med_inputs.get("glyburide-metformin", "No"),
        "glipizide-metformin": med_inputs.get("glipizide-metformin", "No"),
        "glimepiride-pioglitazone": med_inputs.get("glimepiride-pioglitazone", "No"),
        "metformin-rosiglitazone": med_inputs.get("metformin-rosiglitazone", "No"),
        "metformin-pioglitazone": med_inputs.get("metformin-pioglitazone", "No"),
    }

    df_row, df_cat = build_feature_row(inputs)
    X = align_to_training(df_row, df_cat, scaler)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]
    label = "⚠️ HIGH RISK — Likely Readmitted" if prob >= threshold else "✅ LOW RISK — Unlikely Readmitted"

    st.markdown("---")
    st.subheader("Prediction Result")

    col_a, col_b = st.columns(2)
    col_a.metric("Readmission Probability", f"{prob:.1%}")
    col_b.metric("Prediction", label)

    # Risk bar
    bar_color = "#e74c3c" if prob >= threshold else "#2ecc71"
    st.markdown(
        f"""
        <div style="background:#ddd;border-radius:8px;height:24px;width:100%">
          <div style="background:{bar_color};border-radius:8px;height:24px;width:{prob*100:.1f}%;
                      display:flex;align-items:center;padding-left:8px;color:white;font-weight:bold">
            {prob:.1%}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("📋 Feature values sent to model"):
        combined = pd.concat(
            [df_row[COLS_NUM + ["age_group", "has_weight"]].reset_index(drop=True),
             df_cat.reset_index(drop=True)],
            axis=1,
        )
        st.dataframe(combined.T.rename(columns={0: "Value"}), use_container_width=True)
