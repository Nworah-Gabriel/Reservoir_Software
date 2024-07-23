from django.db import models
from datetime import datetime
from uuid import uuid4

class ProductionData(models.Model):
    date = models.DateTimeField(default=datetime.now())
    well_id =  models.UUIDField(default=uuid4)
    production_rate = models.FloatField()
    pressure = models.FloatField()
    temperature = models.FloatField()

    def __str__(self):
        return f"Production data for {self.well_id} on {self.date}"

class EquipmentData(models.Model):
    equipment_id = models.CharField(default="", max_length=100, primary_key=True)
    well_id = models.UUIDField(default=uuid4)
    installation_date = models.DateTimeField(default=datetime.now())
    status = models.CharField(max_length=20, default="")
    last_maintenance_date = models.DateTimeField(default=datetime.now())

    def __str__(self):
       return f"Equipment {self.equipment_id} status {self.status}"


class MaintenanceLog(models.Model):
    maintenance_date = models.DateTimeField(default=datetime.now())
    equipment_id = models.CharField(max_length=100, default="")
    description = models.TextField(default="")
    cost = models.FloatField(default=0.00)

    def __str__(self):
        return f"Maintenance on {self.equipment.equipment_id} on {self.maintenance_date}"