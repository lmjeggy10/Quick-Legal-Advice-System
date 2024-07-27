import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models and encoders
with open('clf_severity.pkl', 'rb') as f:
    clf_severity = pickle.load(f)

with open('reg_imprisonment.pkl', 'rb') as f:
    reg_imprisonment = pickle.load(f)

with open('reg_fine.pkl', 'rb') as f:
    reg_fine = pickle.load(f)

with open('clf_advice.pkl', 'rb') as f:
    clf_advice = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

label_encoders = {}
for column in ['VictimGender', 'Location', 'CrimeType', 'CrimeSeriousness', 'LegalAdvice']:
    with open(f'label_encoder_{column}.pkl', 'rb') as f:
        label_encoders[column] = pickle.load(f)

# Streamlit app
st.title('Quick Legal Advice System')

# User inputs
victim_age = st.number_input('Victim Age', min_value=18, max_value=100, value=30)
victim_gender = st.selectbox('Victim Gender', ['Male', 'Female'])
location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])
crime_type = st.selectbox('Crime Type', ['Offences Against the Human Body', 'Offences Against Property', 
                                         'Offences Relating to Public Tranquillity', 'Offences Relating to Document', 
                                         'Offences Against Women and Children', 'Offences Against the State and Terrorism', 
                                         'Offences Relating to Elections'])

# Preprocess inputs
input_data = pd.DataFrame([[victim_age, victim_gender, location, crime_type]], 
                          columns=['VictimAge', 'VictimGender', 'Location', 'CrimeType'])

for column in ['VictimGender', 'Location', 'CrimeType']:
    input_data[column] = label_encoders[column].transform(input_data[column])

input_data = scaler.transform(input_data)

# Predictions
prediction_severity = clf_severity.predict(input_data)
prediction_imprisonment = reg_imprisonment.predict(input_data)
prediction_fine = reg_fine.predict(input_data)
prediction_advice = clf_advice.predict(input_data)

# Decode predictions
severity = label_encoders['CrimeSeriousness'].inverse_transform(prediction_severity)[0]
advice = label_encoders['LegalAdvice'].inverse_transform(prediction_advice)[0]

# Display results
st.write(f"Predicted Crime Seriousness: {severity}")
st.write(f"Predicted Imprisonment (months): {int(prediction_imprisonment[0])}")
st.write(f"Predicted Fine (INR): {int(prediction_fine[0])}")
st.write(f"Legal Advice: {advice}")
