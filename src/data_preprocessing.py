# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing steps:
    - Handle missing values (if any)
    - Convert data types if necessary
    - Scale/normalize features (if required)
    """
    # For the Credit Card Fraud dataset, there are no missing values.
    # However, if needed, you can add imputation steps here.
    # Also note that most features are already scaled (via PCA).
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    filepath = '../data/creditcard.csv'
    df = load_data(filepath)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
