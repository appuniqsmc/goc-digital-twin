import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("Onco ICU Goals of Care Digital Twin")

# =========================
# MODEL CREATION
# =========================
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

    data["target_GoC"] = np.random.choice(
        ["Full Care","Limited Care","Comfort Care"], n
    )

    X = data.drop(columns=["target_GoC"])
    y = data["target_GoC"]

    X_enc = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_enc, y)

    return model, X_enc.columns

model, feature_columns = create_and_train()

# =========================
# TABS UI
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Simulation",
    "Case Bank",
    "Explainability"
])

# =========================
# TAB 1 — PREDICTION
# =========================
with tab1:

    st.header("Clinical Prediction")

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

    if st.button("Predict GoC"):

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

        for c,p in zip(model.classes_, prob):
            st.write(f"{c}: {round(p*100,2)}%")

# =========================
# TAB 2 — SIMULATION
# =========================
with tab2:

    st.header("What-If Simulation")

    sim_family = st.selectbox(
        "Change Family Preference",
        ["Aggressive","Comfort","Undecided"]
    )

    if st.button("Run Simulation"):

        sim_patient = patient.copy()
        sim_patient["Family_Preference"] = sim_family

        sim_enc = pd.get_dummies(sim_patient)
        sim_enc = sim_enc.reindex(columns=feature_columns, fill_value=0)

        sim_pred = model.predict(sim_enc)[0]

        st.info(f"Simulated Outcome: {sim_pred}")

# =========================
# TAB 3 — CASE BANK
# =========================
with tab3:

    st.header("Teaching Case Bank")

    cases = {
        "Metastatic Cancer Poor Prognosis": {
            "Age":70,"GCS":8,"Charlson_Comorbidity_Index":7,
            "Diagnosis":"Sepsis","Mechanical_Ventilation":"Yes",
            "Active_Malignancy":"Progressive","Clinical_Prognosis":"Poor",
            "Family_Preference":"Comfort","ICU_Day":5,"Physician_Level":"Specialist"
        },

        "Young Sepsis Reversible": {
            "Age":45,"GCS":14,"Charlson_Comorbidity_Index":1,
            "Diagnosis":"Sepsis","Mechanical_Ventilation":"No",
            "Active_Malignancy":"Controlled","Clinical_Prognosis":"Good",
            "Family_Preference":"Aggressive","ICU_Day":2,"Physician_Level":"Resident"
        }
    }

    case_select = st.selectbox("Select Case", list(cases.keys()))

    if st.button("Run Case"):

        case_patient = pd.DataFrame([cases[case_select]])

        case_enc = pd.get_dummies(case_patient)
        case_enc = case_enc.reindex(columns=feature_columns, fill_value=0)

        case_pred = model.predict(case_enc)[0]

        st.success(f"Case Prediction: {case_pred}")

# =========================
# TAB 4 — EXPLAINABILITY
# =========================
with tab4:

    st.header("Decision Drivers")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = plt.figure()
    plt.barh(importance["Feature"][:10], importance["Importance"][:10])
    plt.gca().invert_yaxis()
    st.pyplot(fig)





