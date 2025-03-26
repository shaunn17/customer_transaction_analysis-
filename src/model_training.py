# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features

def train_models(X_train, y_train):
    """
    Train both Logistic Regression and Random Forest classifiers.
    """
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and return precision.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return report, precision

if __name__ == '__main__':
    # Load and preprocess data
    filepath = '../data/creditcard.csv'
    df = load_data(filepath)
    df = preprocess_data(df)
    df = add_features(df)
    
    # Split the data (ensure new features are included)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_report, lr_precision = evaluate_model(lr_model, X_test, y_test)
    print("Logistic Regression Report:\n", lr_report)
    print("Logistic Regression Precision:", lr_precision)
    
    # Evaluate Random Forest
    rf_report, rf_precision = evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Report:\n", rf_report)
    print("Random Forest Precision:", rf_precision)
    
    # Assuming we choose the model with best precision for deployment.
