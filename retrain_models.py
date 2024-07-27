import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

# Define possible values for each column
locations = ['Urban', 'Suburban', 'Rural']
crime_types = ['Offences Against the Human Body', 'Offences Against Property', 'Offences Relating to Public Tranquillity',
               'Offences Relating to Document', 'Offences Against Women and Children', 'Offences Against the State and Terrorism',
               'Offences Relating to Elections']
crime_severities = ['Low', 'Medium', 'High']
advice_given = ['Seek immediate legal counsel', 'File a police report', 'Attend a mediation session',
                'Consult a lawyer', 'Seek protection order', 'Contact national authorities', 'Report to election commission']

# Generate random data
np.random.seed(42)
random.seed(42)
data = {
    'VictimAge': np.random.randint(18, 65, size=1000),
    'VictimGender': np.random.choice(['Male', 'Female'], size=1000),
    'Location': np.random.choice(locations, size=1000),
    'CrimeType': np.random.choice(crime_types, size=1000),
    'CrimeSeriousness': np.random.choice(crime_severities, size=1000),
    'Imprisonment': np.random.randint(1, 240, size=1000),  # up to 20 years of imprisonment
    'Fine': np.random.randint(1000, 1000000, size=1000),  # fine between 1000 and 1,000,000
    'LegalAdvice': np.random.choice(advice_given, size=1000)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
label_encoders = {}
for column in ['VictimGender', 'Location', 'CrimeType', 'CrimeSeriousness', 'LegalAdvice']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])
    with open(f'label_encoder_{column}.pkl', 'wb') as f:
        pickle.dump(label_encoders[column], f)

# Split the dataset into features and target variables
X = df[['VictimAge', 'VictimGender', 'Location', 'CrimeType']]
y_severity = df['CrimeSeriousness']
y_imprisonment = df['Imprisonment']
y_fine = df['Fine']
y_advice = df['LegalAdvice']

# Split into training and testing sets
X_train, X_test, y_train_severity, y_test_severity = train_test_split(X, y_severity, test_size=0.2, random_state=42)
_, _, y_train_imprisonment, y_test_imprisonment = train_test_split(X, y_imprisonment, test_size=0.2, random_state=42)
_, _, y_train_fine, y_test_fine = train_test_split(X, y_fine, test_size=0.2, random_state=42)
_, _, y_train_advice, y_test_advice = train_test_split(X, y_advice, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train a classifier for CrimeSeriousness
clf_severity = RandomForestClassifier(random_state=42)
clf_severity.fit(X_train, y_train_severity)
y_pred_severity = clf_severity.predict(X_test)
print(f"CrimeSeriousness Accuracy: {accuracy_score(y_test_severity, y_pred_severity)}")

# Train a regressor for Imprisonment
reg_imprisonment = RandomForestRegressor(random_state=42)
reg_imprisonment.fit(X_train, y_train_imprisonment)
y_pred_imprisonment = reg_imprisonment.predict(X_test)
print(f"Imprisonment MSE: {mean_squared_error(y_test_imprisonment, y_pred_imprisonment)}")

# Train a regressor for Fine
reg_fine = RandomForestRegressor(random_state=42)
reg_fine.fit(X_train, y_train_fine)
y_pred_fine = reg_fine.predict(X_test)
print(f"Fine MSE: {mean_squared_error(y_test_fine, y_pred_fine)}")

# Train a classifier for LegalAdvice
clf_advice = RandomForestClassifier(random_state=42)
clf_advice.fit(X_train, y_train_advice)
y_pred_advice = clf_advice.predict(X_test)
print(f"LegalAdvice Accuracy: {accuracy_score(y_test_advice, y_pred_advice)}")

# Save models
with open('clf_severity.pkl', 'wb') as f:
    pickle.dump(clf_severity, f)

with open('reg_imprisonment.pkl', 'wb') as f:
    pickle.dump(reg_imprisonment, f)

with open('reg_fine.pkl', 'wb') as f:
    pickle.dump(reg_fine, f)

with open('clf_advice.pkl', 'wb') as f:
    pickle.dump(clf_advice, f)
