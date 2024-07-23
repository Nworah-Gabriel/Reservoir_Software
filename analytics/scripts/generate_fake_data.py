import pandas as pd
import numpy as np
from faker import Faker
import random
import uuid
from datetime import datetime, timezone

fake = Faker()

# Function to generate fake production data
def generate_production_data(num_records=100):
    data = {
        'date': [fake.date_time_this_decade(tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f%z') for _ in range(num_records)],
        'well_id': [str(uuid.uuid4()) for _ in range(num_records)],
        'production_rate': [round(random.uniform(100, 1000), 2) for _ in range(num_records)],
        'pressure': [round(random.uniform(1000, 5000), 2) for _ in range(num_records)],
        'temperature': [round(random.uniform(50, 150), 2) for _ in range(num_records)]
    }
    return pd.DataFrame(data)

# Function to generate fake equipment data
def generate_equipment_data(num_records=100):
    data = {
        'equipment_id': [str(uuid.uuid4()) for _ in range(num_records)],
        'well_id': [str(uuid.uuid4())for _ in range(num_records)],
        'installation_date': [fake.date_time_this_decade(tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f%z') for _ in range(num_records)],
        'status': [random.choice(['operational', 'under maintenance', 'failed']) for _ in range(num_records)],
        'last_maintenance_date': [fake.date_time_this_year(tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f%z') for _ in range(num_records)]
    }
    return pd.DataFrame(data)

# Function to generate fake maintenance log data
def generate_maintenance_logs(num_records=100):
    data = {
        'maintenance_date': [fake.date_time_this_year(tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f%z') for _ in range(num_records)],
        'equipment_id': [str(uuid.uuid4()) for _ in range(num_records)],
        'description': [fake.sentence(nb_words=6) for _ in range(num_records)],
        'cost': [round(random.uniform(1000, 10000), 2) for _ in range(num_records)]
    }
    return pd.DataFrame(data)

# Generate datasets
production_data = generate_production_data()
equipment_data = generate_equipment_data()
maintenance_logs = generate_maintenance_logs()

# Save datasets to CSV files
production_data.to_csv('production_data.csv', index=False)
equipment_data.to_csv('equipment_data.csv', index=False)
maintenance_logs.to_csv('maintenance_logs.csv', index=False)

print("Datasets generated and saved to CSV files.")
