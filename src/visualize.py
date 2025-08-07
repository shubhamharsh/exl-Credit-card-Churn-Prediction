import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def visualize_data(df, plots_dir, prefix=""):
    # Set style
    sns.set_style("whitegrid")
    
    # Box plots by Churn
    numerical_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Churn', y=col, data=df)
        plt.title(f'{col} by Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}boxplots_by_churn.png'))
    plt.close()

def visualize_model_performance(model, X_test, y_test, plots_dir):
    # Predict
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save classification report as text
    cr = classification_report(y_test, y_pred)
    with open(os.path.join(plots_dir, 'classification_report.txt'), 'w') as f:
        f.write(cr)