from django.core.management.base import BaseCommand
from data_entry.models import ProductionData, EquipmentData, MaintenanceLog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class Command(BaseCommand):
    help = 'Train the predictive models using data from the database'

    def handle(self, *args, **kwargs):
        # Fetch data from the database
        production_data = ProductionData.objects.all().values()
        equipment_data = EquipmentData.objects.all().values()
        maintenance_logs = MaintenanceLog.objects.all().values()

        # Convert to DataFrame
        production_df = pd.DataFrame(production_data)
        equipment_df = pd.DataFrame(equipment_data)
        maintenance_logs_df = pd.DataFrame(maintenance_logs)

        # Example: Train a model using production data
        X = production_df[['production_rate', 'pressure', 'temperature']]
        y = production_df['well_id']  # Example target variable, adjust as necessary

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, 'analytics/scripts/maintenance_model.pkl')

        self.stdout.write(self.style.SUCCESS('Successfully trained and saved the model'))
