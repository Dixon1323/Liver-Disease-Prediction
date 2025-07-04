import streamlit as st
import joblib
import pandas as pd

# Load the trained Decision Tree model
model = joblib.load("dt_model.joblib")
stage=""

# Define feature columns
feature_columns = [
    "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT",
    "Tryglicerides", "Platelets", "Prothrombin"
]

# Mapping for categorical values
status_map = {"C": 0, "CL": 1, "D": 2}
sex_map = {"M": 1, "F": 0}
ascites_map = {"N": 0, "Y": 1}
hepatomegaly_map = {"N": 0, "Y": 1}
spiders_map = {"N": 0, "Y": 1}
edema_map = {"N": 0, "S": 1, "Y": 2}
drug_map = {"D-penicillamine": 1, "Placebo": 0}

# Streamlit UI
st.title("Cirrhosis Stage Prediction App")
st.write("Enter patient details to predict the cirrhosis stage.")

# User input fields
status = st.selectbox("Status", list(status_map.keys()))
drug = st.selectbox("Drug", list(drug_map.keys()))
age = st.number_input("Age in Days", min_value=1, max_value=40000, value=18250)
sex = st.selectbox("Sex", list(sex_map.keys()))
ascites = st.selectbox("Ascites", list(ascites_map.keys()))
hepatomegaly = st.selectbox("Hepatomegaly", list(hepatomegaly_map.keys()))
spiders = st.selectbox("Spiders", list(spiders_map.keys()))
edema = st.selectbox("Edema", list(edema_map.keys()))

bilirubin = st.number_input("Bilirubin", min_value=0.1, max_value=50.0, value=0.3)
cholesterol = st.number_input("Cholesterol", min_value=50.0, max_value=1000.0, value=200.0)
albumin = st.number_input("Albumin", min_value=0.5, max_value=5.0, value=2.96)
copper = st.number_input("Copper", min_value=0.0, max_value=500.0, value=84.0)
alk_phos = st.number_input("Alkaline Phosphatase", min_value=10.0, max_value=5000.0, value=1500.8)
sgot = st.number_input("SGOT", min_value=5.0, max_value=500.0, value=99.43)
Tryglicerides = st.number_input("Tryglicerides", min_value=10.0, max_value=500.0, value=109.0)
platelets = st.number_input("Platelets", min_value=50.0, max_value=500.0, value=292.0)
prothrombin = st.number_input("Prothrombin", min_value=5.0, max_value=20.0, value=10.2)

# Predict button
if st.button("Predict Stage"):
    # Convert categorical inputs to numerical
    user_data = {
        "Status": status_map[status],
        "Drug": drug_map[drug],
        "Age": age,
        "Sex": sex_map[sex],
        "Ascites": ascites_map[ascites],
        "Hepatomegaly": hepatomegaly_map[hepatomegaly],
        "Spiders": spiders_map[spiders],
        "Edema": edema_map[edema],
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": Tryglicerides,
        "Platelets": platelets,
        "Prothrombin": prothrombin
    }
    
    # Convert to DataFrame
    test_df = pd.DataFrame([user_data])
    
    # Predict the stage
    prediction = model.predict(test_df)[0]
    prediction_labels = {
    1.0: "Healthy Liver",
    2.0: "Fatty Liver",
    3.0: "Fibrosis of Liver",
    4.0: "Cirrhosis of Liver"
    }
    predicted_label = prediction_labels.get(prediction, "Unknown Condition")


    
    
    # Display result
    st.success(f"Predicted Condition: **{predicted_label}**")
