import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Feature definitions (mirrors notebook exactly)
# ─────────────────────────────────────────────
COLS_NUM = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

COLS_CAT = [
    'race', 'gender', 'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code'
]

COLS_CAT_NUM = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

TOP_10_SPEC = [
    'UNK', 'InternalMedicine', 'Emergency/Trauma',
    'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
    'Nephrology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Radiologist'
]

AGE_MAP = {
    '[0-10)': 0, '[10-20)': 10, '[20-30)': 20, '[30-40)': 30,
    '[40-50)': 40, '[50-60)': 50, '[60-70)': 60,
    '[70-80)': 70, '[80-90)': 80, '[90-100)': 90
}

# ─────────────────────────────────────────────
# Preprocessing (mirrors notebook pipeline)
# ─────────────────────────────────────────────

def preprocess_row(inputs: dict, col2use: list) -> pd.DataFrame:
    """Turn UI inputs into the same feature vector the model was trained on."""
    row = inputs.copy()

    # Replace missing markers
    for k in ['race', 'payer_code', 'medical_specialty']:
        if not row.get(k) or row[k] == '?':
            row[k] = 'UNK'

    # med_spec bucketing
    row['med_spec'] = row['medical_specialty'] if row['medical_specialty'] in TOP_10_SPEC else 'Other'

    # age numeric
    row['age_group'] = AGE_MAP.get(row.get('age', '[50-60)'), 50)

    # has_weight
    row['has_weight'] = 1 if row.get('weight') and row['weight'] != '?' else 0

    # Cast cat-num cols to str
    for c in COLS_CAT_NUM:
        row[c] = str(row.get(c, '1'))

    # Build a single-row DataFrame
    df = pd.DataFrame([row])

    # One-hot encode categorical columns
    all_cat_cols = COLS_CAT + COLS_CAT_NUM + ['med_spec']
    df_cat = pd.get_dummies(df[all_cat_cols], drop_first=True)

    # Combine
    df_num = df[COLS_NUM + ['age_group', 'has_weight']].copy()
    df_full = pd.concat([df_num.reset_index(drop=True),
                          df_cat.reset_index(drop=True)], axis=1)

    # Align to training columns
    for c in col2use:
        if c not in df_full.columns:
            df_full[c] = 0
    df_full = df_full[col2use]

    return df_full


# ─────────────────────────────────────────────
# Model loading / training
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    """
    Try to load pre-trained model + scaler + column list from pickle files.
    If they don't exist, train a quick model on the diabetic_data.csv if available,
    otherwise fall back to a lightweight demo model.
    """
    scaler_path  = "pickle-files/scaler.sav"
    model_path   = "pickle-files/best_classifier.pkl"
    cols_path    = "pickle-files/col2use.pkl"

    if os.path.exists(scaler_path) and os.path.exists(model_path) and os.path.exists(cols_path):
        scaler  = pickle.load(open(scaler_path, 'rb'))
        model   = pickle.load(open(model_path,  'rb'))
        col2use = pickle.load(open(cols_path,   'rb'))
        return model, scaler, col2use, "pre-trained"

    # ── Train from CSV if available ──────────────────────────────────────
    csv_candidates = ["diabetic_data.csv", "data/diabetic_data.csv"]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if csv_path:
        return _train_from_csv(csv_path)

    # ── Demo fallback ─────────────────────────────────────────────────────
    return _demo_model()


def _train_from_csv(csv_path):
    """Full pipeline matching the notebook."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, roc_auc_score

    df = pd.read_csv(csv_path)
    df = df.replace('?', np.nan)
    df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
    df['readmission_status'] = (df.readmitted == '<30').astype('int')

    for c in ['race', 'payer_code', 'medical_specialty']:
        df[c] = df[c].fillna('UNK')

    df['med_spec'] = df['medical_specialty'].copy()
    df.loc[~df.med_spec.isin(TOP_10_SPEC), 'med_spec'] = 'Other'

    df[COLS_CAT_NUM] = df[COLS_CAT_NUM].astype('str')
    df_cat = pd.get_dummies(df[COLS_CAT + COLS_CAT_NUM + ['med_spec']], drop_first=True)

    df['age_group']  = df.age.replace(AGE_MAP)
    df['has_weight'] = df.weight.notnull().astype('int')

    cols_extra = ['age_group', 'has_weight']
    col2use = COLS_NUM + list(df_cat.columns) + cols_extra

    enc_df = pd.concat([df, df_cat], axis=1)
    df_data = enc_df[col2use + ['readmission_status']].dropna()

    df_data = df_data.sample(frac=1, random_state=42).reset_index(drop=True)
    df_valid_test = df_data.sample(frac=0.30, random_state=42)
    df_train_all  = df_data.drop(df_valid_test.index)

    rows_pos  = df_train_all.readmission_status == 1
    df_train  = pd.concat([
        df_train_all.loc[rows_pos],
        df_train_all.loc[~rows_pos].sample(n=rows_pos.sum(), random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = df_train[col2use].values
    y_train = df_train['readmission_status'].values
    X_all   = df_train_all[col2use].values

    scaler = StandardScaler()
    scaler.fit(X_all)
    X_train_tf = scaler.transform(X_train)

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    random_grid = {
        'n_estimators': range(100, 300, 100),
        'max_depth':    range(2, 5, 1),
        'learning_rate':[0.01, 0.1]
    }
    auc_score = make_scorer(roc_auc_score)
    gbc_random = RandomizedSearchCV(gbc, random_grid, n_iter=6, cv=2,
                                    scoring=auc_score, random_state=42, verbose=0)
    gbc_random.fit(X_train_tf, y_train)
    best_model = gbc_random.best_estimator_

    os.makedirs("pickle-files", exist_ok=True)
    pickle.dump(scaler,     open("pickle-files/scaler.sav",         'wb'))
    pickle.dump(best_model, open("pickle-files/best_classifier.pkl",'wb'), protocol=4)
    pickle.dump(col2use,    open("pickle-files/col2use.pkl",        'wb'))

    return best_model, scaler, col2use, "trained on upload"


def _demo_model():
    """Tiny demo model with a representative feature set."""
    np.random.seed(42)
    n = 2000
    col2use = COLS_NUM + ['age_group', 'has_weight']

    X = np.random.randn(n, len(col2use))
    y = (X[:, 6] + np.random.randn(n) * 0.5 > 0).astype(int)  # number_inpatient is key

    scaler = StandardScaler()
    X_tf = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_tf, y)

    return model, scaler, col2use, "demo"


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

MED_OPTIONS  = ['No', 'Steady', 'Up', 'Down']
RACE_OPTIONS = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'UNK']
GENDER_OPT   = ['Male', 'Female', 'Unknown/Invalid']
AGE_OPTIONS  = list(AGE_MAP.keys())
GLU_OPTIONS  = ['None', 'Norm', '>200', '>300']
A1C_OPTIONS  = ['None', 'Norm', '>7', '>8']
CHG_OPTIONS  = ['No', 'Ch']

ADMISSION_TYPE = {
    '1-Emergency': '1', '2-Urgent': '2', '3-Elective': '3',
    '4-Newborn': '4', '5-Not Available': '5', '6-NULL': '6',
    '7-Trauma Center': '7', '8-Not Mapped': '8'
}
DISCHARGE_DISP = {
    '1-Discharged to home': '1', '2-Short-term hospital': '2',
    '3-SNF': '3', '4-ICF': '4', '6-Home health service': '6',
    '7-AMA': '7', '8-Home IV provider': '8', '9-Admitted as inpatient': '9',
    '10-Neonate transferred': '10', '12-Still patient': '12',
    '15-Swing Bed': '15', '16-Outreach': '16', '17-Another inpatient': '17',
    '22-Rehab facility': '22', '23-Long term care': '23',
    '24-Nursing facility-Medicare': '24', '25-Not Mapped': '25',
    '27-Federal health care': '27', '28-Psychiatric hospital': '28',
    '29-Critical access hospital': '29', '30-Another institution': '30'
}
ADMISSION_SRC = {
    '1-Physician Referral': '1', '2-Clinic Referral': '2',
    '3-HMO Referral': '3', '4-Transfer from hosp': '4',
    '5-Transfer from SNF': '5', '6-Transfer from other': '6',
    '7-Emergency Room': '7', '8-Court/Law Enforcement': '8',
    '9-Not Available': '9', '10-Transfer from critical': '10',
    '11-Normal Delivery': '11', '12-Premature Delivery': '12',
    '13-Sick Baby': '13', '14-Extramural Birth': '14',
    '15-Not Available': '15', '17-NULL': '17',
    '18-Transfer another health care': '18', '19-Readmission': '19',
    '20-Not Mapped': '20', '21-Unknown': '21',
    '22-Transfer within institution': '22', '23-Born inside': '23',
    '24-Born outside': '24', '25-Transfer from ambulatory': '25',
    '26-Transfer from hospice': '26'
}

MED_SPEC_OPTIONS = TOP_10_SPEC + ['Other']
PAYER_OPTIONS = ['MC', 'MD', 'HM', 'UN', 'BC', 'SP', 'CP', 'SI', 'DM',
                 'CH', 'PO', 'WC', 'OT', 'OG', 'MP', 'FR', 'UNK']



# ─────────────────────────────────────────────
# Polished Streamlit UI helpers
# ─────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(14,165,233,.18), transparent 35%),
                    linear-gradient(135deg, #f8fafc 0%, #eef6ff 55%, #f7fffb 100%);
    }
    .block-container { padding-top: 1.25rem; max-width: 1260px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #164e63 55%, #0f766e 100%);
    }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    .hero {
        border-radius: 30px; padding: 2.2rem 2.4rem; color: white;
        background: linear-gradient(120deg, #0f172a, #0e7490 60%, #0f766e);
        box-shadow: 0 24px 65px rgba(15,23,42,.20); margin-bottom: 1.25rem;
    }
    .hero h1 { font-size: 3rem; line-height: 1.03; margin: 0; font-weight: 850; letter-spacing: -.055em; }
    .hero p { color: #dbeafe; font-size: 1.05rem; max-width: 820px; margin-top: .8rem; }
    .pill { display:inline-block; padding:.42rem .75rem; border-radius:999px; background:rgba(255,255,255,.14);
            border:1px solid rgba(255,255,255,.25); font-weight:750; font-size:.82rem; margin-bottom:.85rem; }
    .section-card {
        background: rgba(255,255,255,.88); border: 1px solid rgba(148,163,184,.28);
        border-radius: 24px; padding: 1rem 1.15rem; box-shadow: 0 15px 45px rgba(15,23,42,.08);
        margin: .9rem 0 .8rem 0;
    }
    .section-title { font-size:1.25rem; font-weight:850; color:#0f172a; letter-spacing:-.02em; }
    .section-subtitle { color:#475569; margin-top:.18rem; }
    .step { background:#0f172a; color:white; border-radius:12px; padding:.25rem .58rem; font-size:.74rem; margin-right:.5rem; }
    .tip { background:#ecfeff; border:1px solid #a5f3fc; color:#155e75; border-radius:18px; padding:.85rem 1rem; margin:.35rem 0 1rem 0; }
    .risk-card { border-radius:26px; padding:1.4rem; box-shadow:0 18px 55px rgba(15,23,42,.14); margin:1rem 0; }
    .risk-high { background:linear-gradient(135deg,#fff1f2,#fee2e2); border:1px solid #fecaca; }
    .risk-low { background:linear-gradient(135deg,#ecfdf5,#dcfce7); border:1px solid #bbf7d0; }
    .risk-title { font-size:1.45rem; font-weight:850; color:#0f172a; margin:0; }
    .risk-text { color:#334155; margin-top:.35rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.82); border:1px solid rgba(148,163,184,.25);
        border-radius:20px; padding:1rem; box-shadow:0 12px 35px rgba(15,23,42,.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,.78); border-radius:999px; padding:.65rem 1rem;
        border:1px solid rgba(148,163,184,.28); font-weight:700;
    }
    .stButton > button { border-radius:16px; min-height:3.15rem; font-weight:850; }
    </style>
    """, unsafe_allow_html=True)

def section_header(step, title, subtitle):
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title"><span class="step">{step}</span>{title}</div>
      <div class="section-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def risk_band(prob):
    if prob >= 0.70: return "Very high"
    if prob >= 0.50: return "High"
    if prob >= 0.30: return "Moderate"
    return "Low"

# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main():
    inject_css()
    st.markdown("""
    <div class="hero">
      <div class="pill">Clinical ML dashboard • Streamlit ready</div>
      <h1>Diabetes Readmission Risk Dashboard</h1>
      <p>Use the labeled sections below to enter patient profile, medication, lab, and visit-history information. The app estimates 30-day readmission risk and supports CSV batch scoring.</p>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, col2use, source = load_artifacts()

    with st.sidebar:
        st.markdown("## Dashboard guide")
        st.markdown("1. Complete the three patient tabs\n2. Click **Analyze readmission risk**\n3. Review probability, risk band, and top model signals")
        st.markdown("---")
        st.markdown("### Model")
        st.success("Gradient Boosting Classifier")
        st.info(f"Artifact source: **{source}**")
        st.metric("Features expected", f"{len(col2use):,}")
        st.warning("Decision support only. Do not use as the sole basis for clinical decisions.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Target", "30-day readmission")
    k2.metric("Workflow", "Single patient + CSV")
    k3.metric("Threshold", "50% high risk")

    section_header("STEP 1", "Enter patient information", "Each tab groups related inputs so you know exactly what you are changing.")
    tab1, tab2, tab3 = st.tabs(["👤 Demographics & admission", "💊 Medications", "🧪 Labs & history"])

    with tab1:
        st.markdown('<div class="tip">These inputs describe the patient, admission path, discharge destination, specialty, payer, and length of stay.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Patient identity")
            race = st.selectbox("Race", RACE_OPTIONS, help="Unknown values are treated as UNK.")
            gender = st.selectbox("Gender", GENDER_OPT)
            age = st.selectbox("Age group", AGE_OPTIONS, index=5)
        with c2:
            st.markdown("#### Encounter details")
            admission_type_label = st.selectbox("Admission type", list(ADMISSION_TYPE.keys()))
            admission_src_label = st.selectbox("Admission source", list(ADMISSION_SRC.keys()), index=6)
            discharge_label = st.selectbox("Discharge disposition", list(DISCHARGE_DISP.keys()))
        with c3:
            st.markdown("#### Care context")
            med_spec = st.selectbox("Medical specialty", MED_SPEC_OPTIONS)
            payer_code = st.selectbox("Payer code", PAYER_OPTIONS, index=16)
            has_weight = st.checkbox("Weight was recorded", value=False, help="Only whether weight exists is used.")
        time_in_hospital = st.slider("Length of stay: time in hospital", 1, 14, 4)

    with tab2:
        st.markdown('<div class="tip">Medication fields show whether each drug was not used, steady, increased, or decreased. Start with the summary, then adjust individual medications.</div>', unsafe_allow_html=True)
        st.markdown("#### Medication summary")
        s1, s2, s3 = st.columns(3)
        change = s1.selectbox("Medication change during encounter", CHG_OPTIONS, help="Ch means one or more diabetes medications changed.")
        diabetesMed = s2.selectbox("Diabetes medication prescribed", ['No', 'Yes'])
        num_medications = s3.slider("Total number of medications", 1, 81, 15)

        st.markdown("#### Individual medication status")
        med_names = [
            ('metformin','Metformin'), ('repaglinide','Repaglinide'), ('nateglinide','Nateglinide'),
            ('chlorpropamide','Chlorpropamide'), ('glimepiride','Glimepiride'), ('acetohexamide','Acetohexamide'),
            ('glipizide','Glipizide'), ('glyburide','Glyburide'), ('tolbutamide','Tolbutamide'),
            ('pioglitazone','Pioglitazone'), ('rosiglitazone','Rosiglitazone'), ('acarbose','Acarbose'),
            ('miglitol','Miglitol'), ('troglitazone','Troglitazone'), ('tolazamide','Tolazamide'),
            ('insulin','Insulin'), ('glyburide-metformin','Glyburide-metformin'),
            ('glipizide-metformin','Glipizide-metformin'), ('glimepiride-pioglitazone','Glimepiride-pioglitazone'),
            ('metformin-rosiglitazone','Metformin-rosiglitazone'), ('metformin-pioglitazone','Metformin-pioglitazone'),
        ]
        meds = {}
        med_cols = st.columns(3)
        for i, (key, label) in enumerate(med_names):
            meds[key] = med_cols[i % 3].selectbox(label, MED_OPTIONS, key=f"med_{key}")

    with tab3:
        st.markdown('<div class="tip">Prior utilization and clinical workload can strongly affect the risk score. Update visit counts, labs, procedures, and diagnoses here.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Labs")
            num_lab_procedures = st.slider("Number of lab procedures", 1, 132, 44)
            max_glu_serum = st.selectbox("Max glucose serum result", GLU_OPTIONS)
            A1Cresult = st.selectbox("A1C result", A1C_OPTIONS)
        with c2:
            st.markdown("#### Procedures and diagnoses")
            num_procedures = st.slider("Number of non-lab procedures", 0, 6, 1)
            number_diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
        with c3:
            st.markdown("#### Prior visits in past year")
            number_outpatient = st.slider("Outpatient visits", 0, 42, 0)
            number_emergency = st.slider("Emergency visits", 0, 76, 0)
            number_inpatient = st.slider("Inpatient visits", 0, 21, 0)

    section_header("STEP 2", "Analyze single-patient risk", "Run the model and review a clear risk label, probability, confidence, and feature signals.")
    predict_btn = st.button("Analyze readmission risk", type="primary", use_container_width=True)

    if predict_btn:
        inputs = {
            'race': race, 'gender': gender, 'age': age,
            'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures, 'num_medications': num_medications,
            'number_outpatient': number_outpatient, 'number_emergency': number_emergency,
            'number_inpatient': number_inpatient, 'number_diagnoses': number_diagnoses,
            'max_glu_serum': max_glu_serum, 'A1Cresult': A1Cresult,
            'change': change, 'diabetesMed': diabetesMed, 'payer_code': payer_code,
            'medical_specialty': med_spec, 'admission_type_id': ADMISSION_TYPE[admission_type_label],
            'discharge_disposition_id': DISCHARGE_DISP[discharge_label],
            'admission_source_id': ADMISSION_SRC[admission_src_label],
            'weight': '[75-100)' if has_weight else None, **meds
        }
        try:
            X = preprocess_row(inputs, col2use)
            X_tf = scaler.transform(X.values)
            prob = float(model.predict_proba(X_tf)[0][1])
            pred = int(prob >= 0.5)
            confidence = max(prob, 1 - prob)
            label = "High risk" if pred else "Lower risk"
            card_class = "risk-high" if pred else "risk-low"
            action = "Prioritize discharge planning, follow-up scheduling, medication reconciliation, and care coordination." if pred else "Continue standard discharge planning and routine follow-up guidance."
            st.markdown(f"""
            <div class="risk-card {card_class}">
              <p class="risk-title">{label}: {prob:.1%} readmission probability</p>
              <p class="risk-text">Risk band: <b>{risk_band(prob)}</b>. {action}</p>
            </div>
            """, unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk label", label)
            r2.metric("Probability", f"{prob:.1%}")
            r3.metric("Confidence", f"{confidence:.1%}")
            r4.metric("Risk band", risk_band(prob))
            st.progress(prob, text=f"Readmission probability: {prob:.1%}")

            with st.expander("Top model signals", expanded=True):
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(model.feature_importances_, index=col2use).sort_values(ascending=False).head(12)
                    st.bar_chart(importances)
                    st.caption("These are global feature importances, not a patient-specific explanation.")
                else:
                    st.info("Feature importances are not available for this model object.")

            with st.expander("Input summary used for this prediction"):
                st.json({
                    "demographics": {"race": race, "gender": gender, "age": age},
                    "admission": {"type": admission_type_label, "source": admission_src_label, "discharge": discharge_label, "time_in_hospital": time_in_hospital},
                    "history": {"outpatient": number_outpatient, "emergency": number_emergency, "inpatient": number_inpatient, "diagnoses": number_diagnoses},
                    "labs": {"lab_procedures": num_lab_procedures, "max_glucose": max_glu_serum, "A1C": A1Cresult},
                    "medications": {"total": num_medications, "changed": change, "diabetesMed": diabetesMed},
                })
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

    section_header("STEP 3", "Batch prediction from CSV", "Upload a CSV matching diabetic_data.csv to score multiple patient records.")
    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_up):,} rows and {len(df_up.columns):,} columns.")
        with st.spinner("Preprocessing and predicting batch records..."):
            try:
                df_up = df_up.replace('?', np.nan)
                if 'discharge_disposition_id' in df_up.columns:
                    df_up = df_up.loc[~df_up.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
                for c in ['race', 'payer_code', 'medical_specialty']:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].fillna('UNK')
                df_up['med_spec'] = df_up.get('medical_specialty', 'UNK').copy() if 'medical_specialty' in df_up.columns else 'UNK'
                df_up.loc[~df_up.med_spec.isin(TOP_10_SPEC), 'med_spec'] = 'Other'
                for c in COLS_CAT_NUM:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].astype('str')
                df_cat = pd.get_dummies(df_up[[c for c in COLS_CAT + COLS_CAT_NUM + ['med_spec'] if c in df_up.columns]], drop_first=True)
                df_up['age_group'] = df_up.age.replace(AGE_MAP) if 'age' in df_up.columns else 50
                df_up['has_weight'] = df_up.weight.notnull().astype('int') if 'weight' in df_up.columns else 0
                df_full = pd.concat([
                    df_up[[c for c in COLS_NUM + ['age_group', 'has_weight'] if c in df_up.columns]].reset_index(drop=True),
                    df_cat.reset_index(drop=True)
                ], axis=1)
                for c in col2use:
                    if c not in df_full.columns:
                        df_full[c] = 0
                X_batch = df_full[col2use].fillna(0).values
                probs = model.predict_proba(scaler.transform(X_batch))[:, 1]
                df_up['readmission_probability'] = probs
                df_up['predicted_readmission'] = (probs >= 0.5).astype(int)
                df_up['risk_band'] = pd.cut(probs, bins=[-0.01, .30, .50, .70, 1.01], labels=['Low', 'Moderate', 'High', 'Very high'])

                b1, b2, b3 = st.columns(3)
                b1.metric("Patients scored", f"{len(df_up):,}")
                b2.metric("High risk", f"{int((probs >= 0.5).sum()):,}")
                b3.metric("Average probability", f"{probs.mean():.1%}")

                preview_cols = ['readmission_probability', 'predicted_readmission', 'risk_band']
                if 'patient_nbr' in df_up.columns:
                    preview_cols = ['patient_nbr'] + preview_cols
                st.dataframe(df_up[preview_cols].head(100), use_container_width=True)
                st.download_button("Download full scored CSV", df_up.to_csv(index=False).encode(), "readmission_predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
