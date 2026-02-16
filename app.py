import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

st.title("Onco ICU Goals of Care Digital Twin")

# ---------------------
# DEBUG FILE LIST
# ---------------------
st.write("Files in current directory:")
st.write(os.listdir())

# ---------------------
# LOAD DATA SAFE
# ---------------------
@st.cache_resource
def load_and_train():

    # Try multiple filenames automatically
    possible_files = [
        "data.csv",
        "GoC_Digital_Twin_AutoML_Ready.csv",
        "goc_digital_twin_automl_ready.csv"
    ]

    file_found = None

    for f in possible_files:
        if os.path.exists(f):
            file_found = f
            break

    if file_found is None:
        st.error("CSV FILE NOT FOUND. Upload dataset to GitHub repo root.")
        st.stop()

    st.success(f"Using dataset file: {file_found}")

    data = pd.read_csv(file_found)

    data.columns = data.columns.str.strip()

    target = "target_GoC"

    if target not in data.columns:
        st.error("target_GoC column missing")
        st.stop()

    X = data.drop(columns=[target])
    y = data[target]

    X_enc = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_enc, y)

    return model, X_enc.columns

model, feature_columns = load_and_train()

# ---------------------
# UI
# ---------------------
age = st.slider("Age", 18, 95, 60)
gcs = st.slider("GCS", 3, 15, 12)
cci = st.slider("Charlson Index", 0, 10, 2)

diagnosis = st.selectbox(
    "Diagnosis",
    ["Sepsis","Stroke","Cardiac Arrest","Pneumonia"]
)

vent = st.selectbox("Mechanical Ventilation", ["Yes","No"])

malignancy = st.selectbox(
    "Active Malignancy",
    ["Controlled","Progressive","Newly Diagnosed"]
)

prognosis = st.selectbox(
    "Clinical Prognosis",
    ["Good","Moderate","Poor"]
)

family = st.selectbox(
    "Family Preference",
    ["Aggressive","Comfort","Undecided"]
)

icu_day = st.slider("ICU Day", 1, 20, 3)

physician = st.selectbox(
    "Physician Level",
    ["Resident","Attending","Specialist"]
)

if st.button("Predict Goals of Care"):

    patient = pd.DataFrame({
        "Patient_ID":[999],
        "Age":[age],
        "GCS":[gcs],
        "Charlson_Comorbidity_Index":[cci],
        "Diagnosis":[diagnosis],
        "Prior_ICU_Admission":["No"],
        "Active_Malignancy":[malignancy],
        "Mechanical_Ventilation":[vent],
        "Clinical_Prognosis":[prognosis],
        "Family_Preference":[family],
        "ICU_Day":[icu_day],
        "Physician_Level":[physician]
    })

    patient_enc = pd.get_dummies(patient)
    patient_enc = patient_enc.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(patient_enc)[0]
    prob = model.predict_proba(patient_enc)[0]

    st.subheader("Prediction")
    st.success(pred)

    st.subheader("Probabilities")

    for c,p in zip(model.classes_, prob):
        st.write(f"{c}: {round(p*100,2)}%")


