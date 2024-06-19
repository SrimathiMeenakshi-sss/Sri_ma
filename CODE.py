import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the original dataset
file_path = 'C:/Users/srimathi/Desktop/AIML/ML PROJECTS/RAIN PREDICTION/RAIN.csv'
data = pd.read_csv(file_path)

# Clean the data: remove rows with missing target values and duplicates
df_cleaned = data.dropna(subset=['RainTomorrow'])
df_cleaned = df_cleaned.drop_duplicates()

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

for col in categorical_columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

# Fill missing values
df_cleaned.fillna(df_cleaned.mean(), inplace=True)

# Drop 'Date' and 'RISK_MM' columns # UNWANTED COLUMS
df_cleaned = df_cleaned.drop(columns=['Date', 'RISK_MM'])

# Define features and target variable
X = df_cleaned.drop(columns=['RainTomorrow'])
y = df_cleaned['RainTomorrow']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Instantiate the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Input new data as a dataframe
new_data1 = pd.DataFrame({
    'Date': ['2024-06-01'],  # Just a placeholder, you can add more dates
    'Location': ['Cobar'],
    'MinTemp': [12.3],
    'MaxTemp': [25.7],
    'Rainfall': [0.0],
    'Evaporation': [4.4],
    'Sunshine': [8.2],
    'WindGustDir': ['NW'],
    'WindGustSpeed': [35],
    'WindDir9am': ['NW'],
    'WindDir3pm': ['NW'],
    'WindSpeed9am': [20],
    'WindSpeed3pm': [19],
    'Humidity9am': [600],
    'Humidity3pm': [45],
    'Pressure9am': [1015],
    'Pressure3pm': [1012],
    'Cloud9am': [3],
    'Cloud3pm': [1],
    'Temp9am': [18.5],
    'Temp3pm': [24.1],
    'RainToday': ['No']
})

# Preprocess the new data
for col in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']:
    new_data1[col] = le.fit_transform(new_data1[col])

# Drop 'Date' column from new data
X_new = new_data1.drop(columns=['Date'])

# Add any missing columns with default values
for col in X_train.columns:
    if col not in X_new.columns:
        X_new[col] = 0

# Ensure the order of columns matches the training data
X_new = X_new[X_train.columns]

# Make predictions on the new data
new_predictions = rf.predict(X_new)

# Add predictions to the new data
new_data1['RainTomorrow_Prediction'] = new_predictions

# Convert predictions back to original labels (0: No, 1: Yes)
new_data1['RainTomorrow_Prediction'] = new_data1['RainTomorrow_Prediction'].map({0: 'No', 1: 'Yes'})

print(new_data1)

new_data2 = pd.DataFrame({
    'Date': ['2024-06-01'],  # Just a placeholder, you can add more dates
    'Location': ['Balarat'],
    'MinTemp': [15.1],
    'MaxTemp': [19.1],
    'Rainfall': [6.0],
    'Evaporation': [4.6],
    'Sunshine': [2.5],
    'WindGustDir': ['W'],
    'WindGustSpeed': [35],
    'WindDir9am': ['N'],
    'WindDir3pm': ['NW'],
    'WindSpeed9am': [28],
    'WindSpeed3pm': [31],
    'Humidity9am': [60],
    'Humidity3pm': [80],
    'Pressure9am': [1015],
    'Pressure3pm': [1002],
    'Cloud9am': [7],
    'Cloud3pm': [1],
    'Temp9am': [18],
    'Temp3pm': [15],
    'RainToday': ['Yes']})
for col in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']:
    new_data2[col] = le.fit_transform(new_data2[col])
X_new2 = new_data2.drop(columns=['Date'])
for col in X_train.columns:
    if col not in X_new.columns:
        X_new2[col] = 0
X_new2 = X_new2[X_train.columns]
new_predictions2 = rf.predict(X_new2)
new_data2['RainTomorrow_Prediction'] = new_predictions2
new_data2['RainTomorrow_Prediction'] = new_data2['RainTomorrow_Prediction'].map({0: 'No', 1: 'Yes'})
print(new_data2)