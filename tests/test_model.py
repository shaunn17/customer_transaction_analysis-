# tests/test_model.py

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import add_features
from src.model_training import train_models, evaluate_model

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = '../data/creditcard.csv'
        df = load_data(filepath)
        df = preprocess_data(df)
        df = add_features(df)
        X = df.drop('Class', axis=1)
        y = df['Class']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cls.lr_model, cls.rf_model = train_models(cls.X_train, cls.y_train)

    def test_lr_precision(self):
        _, precision = evaluate_model(self.lr_model, self.X_test, self.y_test)
        self.assertGreaterEqual(precision, 0.90)  # Expected precision threshold

    def test_rf_precision(self):
        _, precision = evaluate_model(self.rf_model, self.X_test, self.y_test)
        self.assertGreaterEqual(precision, 0.90)  # Expected precision threshold

if __name__ == '__main__':
    unittest.main()
