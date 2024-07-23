import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'reservoir_pressure': np.random.uniform(1000, 5000, n_samples),  # in psi
    'temperature': np.random.uniform(50, 200, n_samples),  # in degrees Fahrenheit
    'flow_rate': np.random.uniform(100, 5000, n_samples),  # in barrels per day
    'oil_ratio': np.random.uniform(0.1, 0.9, n_samples),  # oil fraction
    'water_ratio': np.random.uniform(0.1, 0.5, n_samples),  # water fraction
    'gas_ratio': np.random.uniform(0.01, 0.1, n_samples),  # gas fraction
    'pipeline_diameter': np.random.uniform(6, 36, n_samples),  # in inches
    'fluid_viscosity': np.random.uniform(0.1, 1, n_samples),  # in centipoise
    'fluid_density': np.random.uniform(600, 1000, n_samples),  # in kg/m^3
    'environment_temp': np.random.uniform(-10, 50, n_samples),  # in degrees Celsius
    'historical_production': np.random.uniform(500, 20000, n_samples)  # in barrels per day
}

print(data)

# Create DataFrame
df = pd.DataFrame(data)

# Generate synthetic target variable (Flow Assurance Index)
df['flow_assurance_index'] = (
    0.3 * (df['reservoir_pressure'] / 5000) +
    0.2 * (1 - df['temperature'] / 200) +
    0.1 * (df['flow_rate'] / 5000) +
    0.1 * df['oil_ratio'] +
    0.1 * (1 - df['water_ratio']) +
    0.1 * (1 - df['gas_ratio']) +
    0.1 * (df['pipeline_diameter'] / 36)
)

# Introduce random noise to the target variable
noise = np.random.normal(0, 0.02, n_samples)
df['flow_assurance_index'] += noise

# Normalize the target variable to be between 0 and 1
df['flow_assurance_index'] = (df['flow_assurance_index'] - df['flow_assurance_index'].min()) / (df['flow_assurance_index'].max() - df['flow_assurance_index'].min())

# Display the first few rows of the dataset
print(df.head())

# Preprocessing
# Split data into features and target
X = df.drop('flow_assurance_index', axis=1)
y = df['flow_assurance_index']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Flow Assurance Index')
plt.ylabel('Predicted Flow Assurance Index')
plt.title('Actual vs Predicted Flow Assurance Index')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.show()

# Save the model
joblib.dump(model, 'best_flow_assurance_model.pkl')
