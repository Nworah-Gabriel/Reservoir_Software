import csv
from django.core.management.base import BaseCommand
from data_entry.models import ProductionData, EquipmentData, MaintenanceLog
from datetime import datetime

class Command(BaseCommand):
    help = 'Load fake data from CSV files into the database'

    def handle(self, *args, **kwargs):
        self.load_production_data()
        self.load_equipment_data()
        self.load_maintenance_logs()
        self.stdout.write(self.style.SUCCESS('Successfully loaded fake data into the database'))

    def load_production_data(self):
        with open('production_data.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ProductionData.objects.create(
                    date=datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S.%f%z'),
                    well_id=row['well_id'],
                    production_rate=row['production_rate'],
                    pressure=row['pressure'],
                    temperature=row['temperature']
                )

    def load_equipment_data(self):
        with open('equipment_data.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                EquipmentData.objects.create(
                    equipment_id=row['equipment_id'],
                    well_id=row['well_id'],
                    installation_date=datetime.strptime(row['installation_date'], '%Y-%m-%d %H:%M:%S.%f%z'),
                    status=row['status'],
                    last_maintenance_date=datetime.strptime(row['last_maintenance_date'], '%Y-%m-%d %H:%M:%S.%f%z')
                )

    def load_maintenance_logs(self):
        with open('maintenance_logs.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                MaintenanceLog.objects.create(
                    maintenance_date=datetime.strptime(row['maintenance_date'], '%Y-%m-%d %H:%M:%S.%f%z'),
                    equipment_id=row['equipment_id'],
                    description=row['description'],
                    cost=row['cost']
                )
