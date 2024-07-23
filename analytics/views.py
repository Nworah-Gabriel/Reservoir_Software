import datetime
import joblib
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .forms import AssuranceFlowForm
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_RandomForest_maintenance_model.pkl')
flow_assurance_model = joblib.load('best_flow_assurance_model.pkl')

@csrf_exempt
def predict_maintenance_cost(request):
    if request.method == 'POST':
        
            # Extract data from request
            data = request.POST

            # Extract individual fields
            production_rate = float(data.get('production_rate'))
            pressure = float(data.get('pressure'))
            temperature = float(data.get('temperature'))
            maintenance_date_str = data.get('maintenance_date')
            installation_date_str = data.get('installation_date')

            # Convert dates to datetime objects
            maintenance_date = pd.to_datetime(maintenance_date_str)
            installation_date = pd.to_datetime(installation_date_str)

            # Calculate feature values
            maintenance_year = maintenance_date.year
            maintenance_month = maintenance_date.month
            equipment_age = (maintenance_date - installation_date).days

            # Create a DataFrame for prediction
            X_new = pd.DataFrame({
                'production_rate': [production_rate],
                'pressure': [pressure],
                'temperature': [temperature],
                'maintenance_year': [maintenance_year],
                'maintenance_month': [maintenance_month],
                'equipment_age': [equipment_age]
            })

            # Predict using the model
            predicted_cost = model.predict(X_new)[0]

            # Return the result
            return render(request, 'maintenanceresult.html', {'predicted_cost': predicted_cost}, status=200)

        

    # Method not allowed
    return render(request, 'predict.html', status=200)




def predict_assurance_flow(request):
    if request.method == 'POST':
        form = AssuranceFlowForm(request.POST)
        if form.is_valid():
            # Extract data from the form
            data = np.array([
                form.cleaned_data['reservoir_pressure'],
                form.cleaned_data['temperature'],
                form.cleaned_data['flow_rate'],
                form.cleaned_data['oil_ratio'],
                form.cleaned_data['water_ratio'],
                form.cleaned_data['gas_ratio'],
                form.cleaned_data['pipeline_diameter'],
                form.cleaned_data['fluid_viscosity'],
                form.cleaned_data['fluid_density'],
                form.cleaned_data['environment_temp'],
                form.cleaned_data['historical_production']
            ]).reshape(1, -1)

            # Make prediction
            prediction = flow_assurance_model.predict(data)[0]

            # Render the result
            return render(request, 'assurance_result.html', {'prediction': prediction})

    else:
        form = AssuranceFlowForm()

    return render(request, 'assurance_predict.html', {'form': form})