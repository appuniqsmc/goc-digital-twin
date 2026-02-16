import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("Onco ICU Digital Twin — Research Version")

# ================= LOAD CSV MODEL =================
@st.cache_resource
def load_model():

    data = pd.read_csv("GoC_Digital_Twin_AutoML_Ready.csv")
    data.columns = data.columns.str.strip()

    target = "target_GoC"

    X = data.drop(columns=[target])
    y = data[target]

    X_enc = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_enc, y)

    return model, X_enc.columns

model, feature_cols = load_model()

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"Prediction",
"Advanced Simulation",
"ICU Timeline Twin",
"Resident Training",
"Explainable AI"
])

# ================= TAB 1 =================
with tab1:

    st.header("Clinical Prediction")

    age = st.slider("Age",18,95,60)
    gcs = st.slider("GCS",3,15,12)
    cci = st.slider("Charlson Index",0,10,2)
    diag = st.selectbox("Diagnosis",["Sepsis","Stroke","Cardiac Arrest","Pneumonia"])
    vent = st.selectbox("Mechanical Ventilation",["Yes","No"])
    mal = st.selectbox("Active Malignancy",["Controlled","Progressive","Newly Diagnosed"])
    prog = st.selectbox("Prognosis",["Good","Moderate","Poor"])
    fam = st.selectbox("Family Preference",["Aggressive","Comfort","Undecided"])
    day = st.slider("ICU Day",1,20,3)
    phys = st.selectbox("Physician Level",["Resident","Attending","Specialist"])

    if st.button("Predict"):

        p = pd.DataFrame({
            "Age":[age],
            "GCS":[gcs],
            "Charlson_Comorbidity_Index":[cci],
            "Diagnosis":[diag],
            "Mechanical_Ventilation":[vent],
            "Active_Malignancy":[mal],
            "Clinical_Prognosis":[prog],
            "Family_Preference":[fam],
            "ICU_Day":[day],
            "Physician_Level":[phys]
        })

        st.session_state["patient"] = p

        pe = pd.get_dummies(p).reindex(columns=feature_cols,fill_value=0)

        pred = model.predict(pe)[0]
        prob = model.predict_proba(pe)[0]

        st.success(pred)

        for c,v in zip(model.classes_,prob):
            st.write(c,":",round(v*100,2),"%")

# ================= TAB 2 =================
with tab2:

    st.header("Multi Variable Simulation")

    if "patient" not in st.session_state:
        st.warning("Run Prediction First")
    else:

        base = st.session_state["patient"].copy()

        sf = st.selectbox("Family Change",["Same","Aggressive","Comfort","Undecided"])
        sp = st.selectbox("Prognosis Change",["Same","Good","Moderate","Poor"])
        sv = st.selectbox("Vent Change",["Same","Yes","No"])
        sd = st.slider("ICU Day Shift",-5,10,0)

        if st.button("Run Simulation"):

            if sf!="Same": base["Family_Preference"]=sf
            if sp!="Same": base["Clinical_Prognosis"]=sp
            if sv!="Same": base["Mechanical_Ventilation"]=sv

            base["ICU_Day"] = base["ICU_Day"] + sd

            se = pd.get_dummies(base).reindex(columns=feature_cols,fill_value=0)

            pred = model.predict(se)[0]
            prob = model.predict_proba(se)[0]

            st.success(pred)

            for c,v in zip(model.classes_,prob):
                st.write(c,":",round(v*100,2),"%")

# ================= TAB 3 =================
with tab3:

    st.header("ICU Trajectory Twin")

    if "patient" not in st.session_state:
        st.warning("Run Prediction First")
    else:

        base = st.session_state["patient"].copy()

        days = st.slider("Simulate Days Forward",1,10,5)

        trend = []

        for d in range(days):
            temp = base.copy()
            temp["ICU_Day"] = temp["ICU_Day"] + d
            te = pd.get_dummies(temp).reindex(columns=feature_cols,fill_value=0)
            pr = model.predict_proba(te)[0]
            trend.append(pr)

        trend = np.array(trend)

        fig = plt.figure()
        for i,c in enumerate(model.classes_):
            plt.plot(trend[:,i],label=c)

        plt.legend()
        plt.xlabel("ICU Days")
        plt.ylabel("Probability")
        st.pyplot(fig)

# ================= TAB 4 =================
with tab4:

    st.header("Resident Training Mode")

    if "patient" not in st.session_state:
        st.warning("Run Prediction First")
    else:

        guess = st.selectbox("Your GoC Guess",["Full Care","Limited Care","Comfort Care"])

        if st.button("Check Answer"):

            p = st.session_state["patient"]
            pe = pd.get_dummies(p).reindex(columns=feature_cols,fill_value=0)
            true = model.predict(pe)[0]

            if guess == true:
                st.success("Correct")
            else:
                st.error(f"Model Suggests: {true}")

# ================= TAB 5 =================
with tab5:

    st.header("Feature Importance")

    imp = pd.DataFrame({
        "Feature":feature_cols,
        "Importance":model.feature_importances_
    }).sort_values("Importance",ascending=False)

    fig = plt.figure()
    plt.barh(imp["Feature"][:10],imp["Importance"][:10])
    plt.gca().invert_yaxis()
    st.pyplot(fig)







