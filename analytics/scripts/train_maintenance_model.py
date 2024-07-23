import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from lightgbm import LGBMRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
maintenance_logs = pd.read_csv('maintenance_logs.csv')
equipment_data = pd.read_csv('equipment_data.csv')
production_data = pd.read_csv('production_data.csv')

# Data Preprocessing
def preprocess_data(maintenance_logs, equipment_data, production_data):
    maintenance_logs = maintenance_logs.drop_duplicates()
    production_data = production_data.drop_duplicates()
    merged_data1 = pd.merge(maintenance_logs, equipment_data, on='equipment_id')
    merged_data = pd.merge(merged_data1, production_data, on='well_id')
    merged_data['maintenance_year'] = pd.to_datetime(merged_data['maintenance_date']).dt.year
    merged_data['maintenance_month'] = pd.to_datetime(merged_data['maintenance_date']).dt.month
    merged_data['equipment_age'] = (pd.to_datetime(merged_data['maintenance_date']) - pd.to_datetime(merged_data['installation_date'])).dt.days
    features = ['production_rate', 'pressure', 'temperature', 'maintenance_year', 'maintenance_month', 'equipment_age']
    target = 'cost'
    X = merged_data[features]
    y = merged_data[target]
    return X, y

# Preprocess the data
X, y = preprocess_data(maintenance_logs, equipment_data, production_data)

# Handle outliers
def handle_outliers(X, y):
    z_scores = np.abs((X - X.mean()) / X.std())
    filtered_entries = (z_scores < 3).all(axis=1)
    X = X[filtered_entries]
    y = y[filtered_entries]
    return X, y

X, y = handle_outliers(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models for comparison
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    # 'LightGBM': LGBMRegressor()
}

# Hyperparameter grids for each model
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    },
    # 'LightGBM': {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 20, 30, -1],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'num_leaves': [31, 61, 127]
    # }
}

# Initialize dictionaries to store results
best_models = {}
best_params = {}
best_scores = {}
test_mse_scores = {}
test_r2_scores = {}

# Evaluate each model
for model_name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    param_grid = {f'regressor__{key}': value for key, value in param_grids[model_name].items()}
    random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_models[model_name] = random_search.best_estimator_
    best_params[model_name] = random_search.best_params_
    best_scores[model_name] = -random_search.best_score_
    
    y_pred = random_search.best_estimator_.predict(X_test)
    test_mse_scores[model_name] = mean_squared_error(y_test, y_pred)
    test_r2_scores[model_name] = r2_score(y_test, y_pred)
    
    print(f"{model_name} - Best Parameters: {best_params[model_name]}")
    print(f"{model_name} - Best Cross-validation Score (MSE): {best_scores[model_name]}")
    print(f"{model_name} - Test Set Score (MSE): {test_mse_scores[model_name]}")
    print(f"{model_name} - Test Set R² Score: {test_r2_scores[model_name]}\n")

# Residual Analysis
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs True Values')
    plt.show()

best_model_name = max(test_r2_scores, key=test_r2_scores.get)
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)
plot_residuals(y_test, y_pred)

# Feature Importance for the best model
if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
    feature_importance = best_model.named_steps['regressor'].feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance for Maintenance Cost Prediction')
    plt.gca().invert_yaxis()
    plt.show()

# Save the best model
joblib.dump(best_model, f'best_{best_model_name}_maintenance_model.pkl')
