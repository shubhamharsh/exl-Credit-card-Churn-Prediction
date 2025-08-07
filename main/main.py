import os
from src.eda import perform_eda
from src.train import train_model
from src.inference import safe_inference
from src.visualize import visualize_data, visualize_model_performance
from src.utils import load_and_preprocess_data

def main():
    # Define paths
    data_path = "data/file.csv"
    cleaned_data_path = "data/cleaned_file.csv"
    models_dir = "models"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data, save cleaned_file.csv
    df, cleaned_df, X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, cleaned_data_path)
    
    # Perform EDA on original dataset and save plots
    perform_eda(df, plots_dir)
    
    # Visualize cleaned dataset
    visualize_data(cleaned_df, plots_dir, prefix="cleaned_")
    
    # Train model on in-memory preprocessed data
    best_model, scaler = train_model(X_train, y_train, models_dir)
    
    # Visualize model performance
    visualize_model_performance(best_model, X_test, y_test, plots_dir)
    
    # Example inference
    new_input = {
        'CustomerID': 'CUST1055',
        'Gender': 'alien',
        'Age': None,
        'Tenure': 4,
        'Balance': None,
        'NumOfProducts': 3,
        'HasCrCard': 'Maybe',
        'IsActiveMember': '1.0',
        'EstimatedSalary': 70000
    }
    result = safe_inference(new_input, models_dir)
    print("Inference Result:", result)

if __name__ == "__main__":
    main()