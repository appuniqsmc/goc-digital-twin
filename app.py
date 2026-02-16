import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Onco ICU Goals of Care Digital Twin")

# ----------------------
# CREATE DATA INSIDE APP
# ----------------------
@st.cache_resource
def create_and_train():

    np.random.seed(42)

    n = 400

    data = pd.DataFrame({
        "Age": np.random.randint(18, 95, n),
        "GCS": np.random.randint(3, 15, n),
        "Charlson_Comorbidity_Index": np.random.randint(0, 10, n),
        "Diagnosis": np.random.choice(["Sepsis","Stroke","Cardiac Arrest","Pneumonia"], n),
        "Mechanical_Ventilation": np.random.choice(["Yes","No"], n),
        "Active_Malignancy": np.random.choice(["Controlled","Progressive","Newly Diagnosed"], n),
        "Clinical_Prognosis": np.random.choice(["Good","Moderate","Poor"], n),
        "Family_Preference": np.random.choice(["Aggressive","Comfort","Undecided"], n),
        "ICU_Day": np.random.randint(1, 20, n),
        "Physician_Level": np.random.choice(["Resident","Attending","Specialist"], n)
    })

    # Create synthetic GoC outcome
    data["target_GoC"] = np.random.choice(
        ["Full Care","Limited Care","Comfort Care"],
        n
    )

    X = data.drop(columns=["target_GoC"])
    y = data["target_GoC"]

    X_enc = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_enc, y)

    return model, X_enc.columns

model, feature_columns = create_and_train()

# ----------------------
# USER INPUT UI
# ----------------------

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

# ----------------------
# PREDICT
# ----------------------

if st.button("Predict Goals of Care"):

    patient = pd.DataFrame({
        "Age":[age],
        "GCS":[gcs],
        "Charlson_Comorbidity_Index":[cci],
        "Diagnosis":[diagnosis],
        "Mechanical_Ventilation":[vent],
        "Active_Malignancy":[malignancy],
        "Clinical_Prognosis":[prognosis],
        "Family_Preference":[family],
        "ICU_Day":[icu_day],
        "Physician_Level":[physician]
    })

    patient_enc = pd.get_dummies(patient)
    patient_enc = patient_enc.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(patient_enc)[0]
    prob = model.predict_proba(patient_enc)[0]

    st.success(pred)

    st.subheader("Probabilities")

    for c,p in zip(model.classes_, prob):
        st.write(f"{c}: {round(p*100,2)}%")



