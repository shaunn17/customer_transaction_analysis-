# src/feature_engineering.py

import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 15+ fraud-indicative features.
    Here we include example features; in practice, you may
    derive additional features based on domain knowledge.
    """
    # Example Feature 1: Transaction Amount Category
    df['AmountCategory'] = pd.cut(df['Amount'], bins=[-1, 50, 200, 1000, df['Amount'].max()], labels=[0,1,2,3])
    
    # Example Feature 2: Time of Day (assuming 'Time' is in seconds)
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Example Feature 3: Flag for low amount transactions (could be common in fraud)
    df['LowAmountFlag'] = (df['Amount'] < 100).astype(int)
    
    # Additional features can include:
    # - Rolling statistics on transaction amounts (if data is time-ordered)
    # - Ratios between Amount and other numerical features (if available)
    # - Aggregated user-level statistics (if customer IDs exist)
    # - Derived features from PCA components (if any insights can be drawn)
    # For illustration, we generate dummy features
    for i in range(4, 16):
        df[f'dummy_feature_{i}'] = np.random.rand(len(df))
    
    return df

if __name__ == '__main__':
    import sys
    filepath = '../data/creditcard.csv'
    df = pd.read_csv(filepath)
    df = add_features(df)
    print("Features added. Dataframe shape:", df.shape)
