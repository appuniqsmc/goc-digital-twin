import streamlit as st
st.cache_resource.clear()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Onco ICU Goals of Care Digital Twin")

# -----------------------------
# Load & Train Model (No pickle)
# -----------------------------
@st.cache_resource
def load_and_train():
    data = pd.read_csvdata = pd.read_csv("data.csv")

    target = "target_GoC"
    X = data.drop(columns=[target])
    y = data[target]

    X_enc = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_enc, y)

    return model, X_enc.columns

model, feature_columns = load_and_train()

# -----------------------------
# UI Inputs
# -----------------------------
st.header("Enter Patient Details")

age = st.slider("Age", 18, 95, 60)
gcs = st.slider("GCS", 3, 15, 12)
cci = st.slider("Charlson Index", 0, 10, 2)

diagnosis = st.selectbox(
    "Diagnosis",
    ["Sepsis","Stroke","Cardiac Arrest","Pneumonia"]
)

vent = st.selectbox(
    "Mechanical Ventilation",
    ["Yes","No"]
)

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

# -----------------------------
# Prediction
# -----------------------------
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
    for c, p in zip(model.classes_, prob):
        st.write(f"{c}: {round(p*100,2)}%")


