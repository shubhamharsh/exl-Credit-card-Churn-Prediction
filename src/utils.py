import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess_data(file_path, cleaned_file_path):
    # Load original data
    df = pd.read_csv(file_path)
    
    # Drop rows with missing Churn
    df.dropna(subset=['Churn'], inplace=True)
    
    # Fill numeric columns with median
    num_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    cat_cols = ['Gender', 'HasCrCard', 'IsActiveMember']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Standardize Gender
    df['Gender'] = df['Gender'].str.strip().str.lower()
    df['Gender'] = df['Gender'].replace({'male': 'Male', 'female': 'Female'})
    
    # Standardize HasCrCard
    df['HasCrCard'] = df['HasCrCard'].astype(str).str.strip()
    df['HasCrCard'] = df['HasCrCard'].replace({
        '1.0': 1, '0.0': 0, 'Yes': 1, 'No': 0, '2.0': 1
    }).astype(int)
    
    # Standardize IsActiveMember
    df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.strip()
    df['IsActiveMember'] = df['IsActiveMember'].replace({
        '1.0': 1, '0.0': 0, '-1': 0, '-1.0': 0, 'No': 0, 'Yes': 1
    }).astype(int)
    
    # Clean Churn
    df['Churn'] = df['Churn'].astype(str).str.strip()
    df = df[df['Churn'].isin(['0.0', '1.0'])]
    df['Churn'] = df['Churn'].replace({'1.0': 1, '0.0': 0}).astype(int)
    
    # One-hot encode Gender
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    if 'Gender_Male' not in df.columns:
        df['Gender_Male'] = 0
    
    # Filter Age
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
    
    # Drop CustomerID
    df.drop('CustomerID', axis=1, inplace=True)
    
    # Save cleaned dataset
    df.to_csv(cleaned_file_path, index=False)
    
    # Features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return df, df, X_train, X_test, y_train, y_test