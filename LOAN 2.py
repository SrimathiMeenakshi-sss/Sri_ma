import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def calculate_interest_rate(row):
    loan_amount = row['LoanAmount']
    if loan_amount < 100000:
        interest_rate = 0.20
    elif 100000 <= loan_amount < 250000:
        interest_rate = 0.30
    else:
        interest_rate = 0.20

    if row['Gender'] == 'Female':
        interest_rate -= 0.05
    if row['Dependents'] > 3:
        interest_rate -= 0.02
    if row['Education'] == 'Non-Graduate':
        interest_rate -= 0.02
    if row['Self_Employed'] == 'Yes':
        interest_rate += 0.10

    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    if total_income < 6000:
        interest_rate -= 0.05
    elif 6000 <= total_income < 15000:
        interest_rate += 0.05
    else:
        interest_rate += 0.15

    loan_term = row['Loan_Amount_Term']
    if loan_term < 240:
        interest_rate += 0.10
    elif 240 <= loan_term < 350:
        interest_rate += 0.05
    elif 350 <= loan_term < 400:
        interest_rate += 0.03
    elif loan_term > 400:
        interest_rate -= 0.10

    if row['Credit_History'] == 0:
        interest_rate += 0.07

    if row['Property_Area'] == 'Urban':
        interest_rate += 0.02
    elif row['Property_Area'] == 'Rural':
        interest_rate -= 0.20

    total_interest = interest_rate * loan_amount
    loan_eligibility = total_interest <= 0.5 * total_income

    return interest_rate, loan_eligibility





# Load training data
train_csv_file_path = 'C:/Users/srimathi/Desktop/AIML/ML PROJECTS/LOAN APPROVAL/TRAIN DATA.csv'
df_train = pd.read_csv(train_csv_file_path)

# Multiply 'LoanAmount' by 1000 for training data
df_train['LoanAmount'] = df_train['LoanAmount'] * 100

# DATA CLEANING for training data
df_train = df_train.dropna()
df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_train = df_train.drop_duplicates()
df_train = df_train[(np.abs(stats.zscore(df_train.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Apply the function to calculate interest rate for training data
df_train[['Interest_Rate', 'LoanEligibility']] = df_train.apply(calculate_interest_rate, axis=1, result_type='expand')
df_train['Total_Interest'] = df_train['LoanAmount'] * df_train['Interest_Rate']
df_train['ApplicantIncome_Total'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
df_train['Loan_Status'] = np.where(df_train['Total_Interest'] > 0.5 * df_train['ApplicantIncome_Total'], 'Reject', 'Approve')

# Encode categorical variables for training data
df_train_encoded = pd.get_dummies(df_train, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'])

# Prepare feature and target variables for training
X_train = df_train_encoded.drop(columns=['Loan_ID', 'Loan_Status'])
y_train = df_train_encoded['Loan_Status'].apply(lambda x: 1 if x == 'Approve' else 0)  # 1 for Approve, 0 for Reject

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)





# Load and clean test data
test_csv_file_path = 'C:/Users/srimathi/Desktop/AIML/ML PROJECTS/LOAN APPROVAL/test DATA.csv'
df_test = pd.read_csv(test_csv_file_path)

# Multiply 'LoanAmount' by 1000 for test data
df_test['LoanAmount'] = df_test['LoanAmount'] * 100

# DATA CLEANING for test data
df_test = df_test.dropna()
df_test = df_test.fillna(df_test.mean(numeric_only=True))
df_test = df_test.drop_duplicates()
df_test = df_test[(np.abs(stats.zscore(df_test.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Apply the function to calculate interest rate for test data
df_test[['Interest_Rate', 'LoanEligibility']] = df_test.apply(calculate_interest_rate, axis=1, result_type='expand')
df_test['Total_Interest'] = df_test['LoanAmount'] * df_test['Interest_Rate']
df_test['ApplicantIncome_Total'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']
df_test['Loan_Status'] = np.where(df_test['Total_Interest'] > 0.5 * df_test['ApplicantIncome_Total'], 'Reject', 'Approve')

# Encode categorical variables for test data
df_test_encoded = pd.get_dummies(df_test, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'])

# Align test data columns with training data columns
X_test = df_test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict using the trained classifier
y_pred = clf.predict(X_test)

# Add predictions to the test DataFrame
df_test['Predicted_Loan_Status'] = np.where(y_pred == 1, 'Approve', 'Reject')

# Display the predictions
print(df_test[['Loan_ID', 'Predicted_Loan_Status', 'ApplicantIncome_Total', 'Total_Interest' ]])





df_new = pd.DataFrame({
    'LoanAmount': [120000, 80000, 300000],  # Adjusted LoanAmount to match multiplied values
    'Gender': ['Male', 'Female', 'Male'],
    'Dependents': ['2', '3', '1'],
    'Education': ['Graduate', 'Non-Graduate', 'Graduate'],
    'Self_Employed': ['No', 'Yes', 'No'],
    'ApplicantIncome': [1000, 4000, 6000],
    'CoapplicantIncome': [4000, 1000, 3000],
    'Loan_Amount_Term': [360, 360, 2400],
    'Credit_History': [1, 0, 1],
    'Property_Area': ['Urban', 'Rural', 'Semiurban']})

df_new['Interest_Rate'] = df_new.apply(calculate_interest_rate, axis=1)

# Encode categorical variables for new data
df_new_encoded = pd.get_dummies(df_new, columns=['Gender', 'Education', 'Self_Employed', 'Property_Area'])

# Align the new data with the training data (to ensure same number of columns)
X_new = df_new_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions on the new data
y_pred_new = clf.predict(X_new)

# Map predictions back to 'Approve' and 'Reject'
df_new['Loan_Status_Prediction'] = np.where(y_pred_new == 1, 'Approve', 'Reject')

# Display the new DataFrame with predictions
print(df_new[['LoanAmount', 'Dependents', 'Interest_Rate', 'Loan_Status_Prediction']])