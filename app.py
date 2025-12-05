import streamlit as st
import pandas as pd
import pickle 
import numpy as np

# Load the trained model and scaler
with open('svm_rbf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Diabetic Foot Prediction App')
st.write('Enter the patient details below to predict the risk of Diabetic Foot.')

# Input fields for numerical features
st.sidebar.header('Numerical Features')
age = st.sidebar.number_input('Age', min_value=17, max_value=80, value=50)
duration_dm = st.sidebar.number_input('Duration of DM', min_value=1.0, max_value=30.0, value=10.0)
bm1 = st.sidebar.number_input('BM1', min_value=13.46, max_value=30.80, value=23.0)
fbs = st.sidebar.number_input('FBS', min_value=55, max_value=390, value=150)
hba1c = st.sidebar.number_input('HbA1C', min_value=5.27, max_value=18.69, value=8.0)
itlni_elisa = st.sidebar.number_input('ITLNI ELISA', min_value=8.67, max_value=1130.01, value=150.0)
ntn1_elisa = st.sidebar.number_input('NTN1 ELISA', min_value=37.74, max_value=1354.56, value=200.0)
tg = st.sidebar.number_input('TG', min_value=39.0, max_value=550.0, value=120.0)
hdl = st.sidebar.number_input('HDL', min_value=6.77, max_value=75.80, value=35.0)
ldl = st.sidebar.number_input('LDL', min_value=3.50, max_value=273.60, value=95.0)
cholesterol = st.sidebar.number_input('CHOLESTEROL', min_value=27.0, max_value=343.0, value=150.0)
vldl = st.sidebar.number_input('VLDL', min_value=2.14, max_value=110.0, value=13.0)

# Input fields for categorical features
st.sidebar.header('Categorical Features')
gender = st.sidebar.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
family_h_o_dm = st.sidebar.selectbox('Family H/O DM', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
stages = st.sidebar.selectbox('STAGES', [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'Age': age,
    'Duration of DM ': duration_dm,
    'BM1': bm1,
    'FBS': fbs,
    'HbA1C': hba1c,
    'ITLNI ELISA': itlni_elisa,
    'NTN1 ELISA': ntn1_elisa,
    'TG': tg,
    'HDL': hdl,
    'LDL': ldl,
    'CHOLESTEROL': cholesterol,
    'VLDL': vldl,
    'Gender': gender,
    'Family H/O DM': family_h_o_dm,
    'STAGES': stages
}])

# Separate numerical and categorical columns for preprocessing
numerical_cols = [
    'Age', 'Duration of DM ', 'BM1', 'FBS', 'HbA1C', 'ITLNI ELISA',
    'NTN1 ELISA', 'TG', 'HDL', 'LDL', 'CHOLESTEROL', 'VLDL'
]
categorical_cols = ['Gender', 'Family H/O DM', 'STAGES']

# Scale numerical features
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Apply one-hot encoding to match training data columns
# Create a template DataFrame with all possible one-hot encoded columns from training
# This ensures consistency even if not all categories are present in a single input
# Assuming X_final from training had these columns. The order matters for prediction.

# Generate dummy columns for Gender
gender_options = [1, 2]
for g_val in gender_options:
    input_data[f'Gender_{g_val}'] = 0
input_data[f'Gender_{gender}'] = 1

# Generate dummy columns for Family H/O DM
family_ho_dm_options = [0, 1]
for f_val in family_ho_dm_options:
    input_data[f'Family H/O DM_{f_val}'] = 0
input_data[f'Family H/O DM_{family_h_o_dm}'] = 1

# Generate dummy columns for STAGES
stages_options = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
for s_val in stages_options:
    input_data[f'STAGES_{s_val}.0'] = 0
input_data[f'STAGES_{stages}.0'] = 1

# Drop original categorical columns from input_data
input_data = input_data.drop(columns=categorical_cols)

# Ensure column order matches the training data exactly.
# This list should be fixed based on the actual training columns.
expected_columns = [
    'Age', 'Duration of DM ',
    'BM1', 'FBS', 'HbA1C', 'ITLNI ELISA', 'NTN1 ELISA',
    'TG', 'HDL', 'LDL', 'CHOLESTEROL', 'VLDL',
    'Gender_1', 'Gender_2',
    'Family H/O DM_0', 'Family H/O DM_1',
    'STAGES_0.0', 'STAGES_1.0', 'STAGES_2.0', 'STAGES_3.0', 'STAGES_4.0', 'STAGES_5.0'
]

# Reindex input_data to match expected_columns, filling missing with 0 (for categories not present)
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Make prediction
if st.button('Predict Diabetic Foot Risk'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error(f'The patient is predicted to have Diabetic Foot. (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'The patient is predicted NOT to have Diabetic Foot. (Probability: {1-prediction_proba:.2f})')

    st.write('---')
    st.write('Feature values used for prediction:')
    st.write(input_data)

# To run this Streamlit app, save it as `app.py` and run `streamlit run app.py` in your terminal.
