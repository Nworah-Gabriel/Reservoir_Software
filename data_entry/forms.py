from django import forms
from .models import ProductionData, EquipmentData, MaintenanceLog

class ProductionDataForm(forms.ModelForm):
    class Meta:
        model = ProductionData
        fields = '__all__'

class EquipmentDataForm(forms.ModelForm):
    class Meta:
        model = EquipmentData
        fields = '__all__'

class MaintenanceLogForm(forms.ModelForm):
    class Meta:
        model = MaintenanceLog
        fields = '__all__'
