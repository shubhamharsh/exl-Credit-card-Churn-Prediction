from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train, models_dir):
    # Load scaler
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    
    # Define and train GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Save best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'{models_dir}/final.pkl')
    print("Best Parameters:", grid_search.best_params_)
    
    return best_model, scaler