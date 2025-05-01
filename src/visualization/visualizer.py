import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class DataVisualizer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None

    def load_data(self):
        """Load the data."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def plot_churn_distribution(self, output_path):
        """Plot the distribution of churn."""
        if self.data is None:
            print("Please load data first using load_data()")
            return None

        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.data, x='churn')
        plt.title('Distribution of Customer Churn')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.savefig(output_path)
        plt.close()

    def plot_correlation_matrix(self, output_path):
        """Plot the correlation matrix of numerical features."""
        if self.data is None:
            print("Please load data first using load_data()")
            return None

        # Select numerical columns
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.data[numerical_cols].corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_feature_importance(self, model, feature_names, output_path):
        """Plot feature importance from the trained model."""
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    # Example usage
    visualizer = DataVisualizer("../../data/processed/processed_data.csv")
    data = visualizer.load_data()
    if data is not None:
        visualizer.plot_churn_distribution("../../output/churn_distribution.png")
        visualizer.plot_correlation_matrix("../../output/correlation_matrix.png") 