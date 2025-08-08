import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, plots_dir):
    # Set style
    sns.set_style("whitegrid")
    
    # Print unique values for categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'CustomerID':
            unique_vals = df[col].unique()
            print(f"\nColumn: {col} (Type: {df[col].dtype})")
            print(f"Unique Values ({len(unique_vals)}): {unique_vals}")
    
    # Print Churn balance
    print("\nChurn Value Counts:")
    print(df['Churn'].value_counts())
    
    # Print Age description
    print("\nAge Description:")
    print(df['Age'].describe())
    
    # Histogram of numerical features
    numerical_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numerical_histograms.png'))
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig(os.path.join(plots_dir, 'churn_distribution.png'))
    plt.close()