# utils/visualization.py
#I forgot to add in the description so im adding it here - This module provides utility functions for additional visualizations (for exploratory analysis or reporting).



import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(df, feature: str):
    """
    Plot distribution of a feature.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

def plot_confusion_matrix(cm, classes):
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

