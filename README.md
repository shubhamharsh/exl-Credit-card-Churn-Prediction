# Churn Prediction Project
## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.73      | 0.74   | 0.74     | 105     |
| 1     | 0.71      | 0.69   | 0.70     | 95      |

**Accuracy**: 0.72 (72.00%)

| Metric        | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Macro Avg     | 0.72      | 0.72   | 0.72     | 200     |
| Weighted Avg  | 0.72      | 0.72   | 0.72     | 200     |

Welcome to the Churn Prediction Project! This repository contains a machine learning pipeline to predict customer churn and a web application to interact with the model. The project leverages a dataset with customer details to train a predictive model and provides a user-friendly interface for predictions.

## Table of Summary

| **Category**         | **Details**                              |
|-----------------------|------------------------------------------|
| **Project Name**      | Churn Prediction Project                 |
| **Repository** | [https://github.com/shubhamharsh/exl-Credit-card-Churn-Prediction](https://github.com/shubhamharsh/exl-Credit-card-Churn-Prediction.git) |
| **Last Updated**      | August 08, 2025, 01:03 AM IST            |
| **Language/Tools**    | Python, Flask, Pandas, Scikit-learn, Tailwind CSS |
| **Dependencies**      | pandas, numpy, scikit-learn, joblib, Flask |
| **Folder Structure**  | - `churn_prediction_project/`: Training pipeline<br>- `churn_web_app/`: Web application |
| **Model Type**        | Random Forest Classifier (GridSearchCV optimized) |
| **Web App Features**  | User input form, churn prediction, probability display |

## Overview

This project predicts customer churn using a dataset (`data/file.csv`) with features such as `CustomerID`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, and `Churn`. The pipeline includes:

- **Data Preprocessing**: Drops rows with missing `Churn`, fills numeric columns (`Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) with median, categorical columns (`Gender`, `HasCrCard`, `IsActiveMember`) with mode, standardizes values, one-hot encodes `Gender`, filters `Age` to 18-100, drops `CustomerID`, and scales features with `MinMaxScaler`. The cleaned dataset is saved as `cleaned_file.csv`.
- **EDA**: Generates console outputs (unique values for categoricals, `Churn` balance, `Age` description) and visualizations (histograms, correlation heatmap, churn distribution) saved in `plots/`.
- **Model Training**: Trains a Random Forest Classifier with GridSearchCV, optimizing parameters (`n_estimators`, `max_depth`, `min_samples_split`), and saves the best model (`models/final.pkl`) and scaler (`models/scaler.pkl`).
- **Web Application**: A Flask-based app (`churn_web_app/`) with a Tailwind CSS-styled interface allows users to input customer data and view churn predictions with probabilities.

## Steps We Have Done
1. **Project Setup**: Initialized the `churn_prediction_project/` folder with a modular structure for preprocessing, EDA, and training.
2. **Data Preprocessing**: Implemented and tested the preprocessing steps in `src/utils.py` based on the Jupyter Notebook (`test2.ipynb`) code, addressing issues like the `ImportError: cannot import name 'perf'` by correcting directory paths.
3. **EDA Development**: Created `src/eda.py` to perform exploratory data analysis, matching the notebook’s console outputs and adding visualizations saved to `plots/`.
4. **Model Training**: Developed `src/train.py` to train a Random Forest model with GridSearchCV, ensuring the pipeline saves `final.pkl` and `scaler.pkl`.
5. **Web App Creation**: Built a separate `churn_web_app/` folder with `app.py`, styled `index.html` and `result.html` using Tailwind CSS, and added a header with a GitHub link (`https://github.com/username/churn-prediction`).
6. **Git Integration**: Provided instructions to set up Git, initialize repositories for both folders, and prepare for pushing to GitHub.
7. **Documentation**: Created this `README.md` to summarize the project and steps taken.

## Installation

### Prerequisites
- Python 3.8+
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/churn-prediction.git
   cd churn_prediction_project
