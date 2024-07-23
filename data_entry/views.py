from django.shortcuts import render, redirect
from .forms import ProductionDataForm, EquipmentDataForm, MaintenanceLogForm
from rest_framework import viewsets
from .models import ProductionData, EquipmentData, MaintenanceLog
from .serializers import ProductionDataSerializer, EquipmentDataSerializer, MaintenanceLogSerializer


def enter_production_data(request):
    if request.method == 'POST':
        form = ProductionDataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('data_entry:enter_production_data')
    else:
        form = ProductionDataForm()
    return render(request, 'data_entry/enter_production_data.html', {'form': form})

def enter_equipment_data(request):
    if request.method == 'POST':
        form = EquipmentDataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('data_entry:enter_equipment_data')
    else:
        form = EquipmentDataForm()
    return render(request, 'data_entry/enter_equipment_data.html', {'form': form})

def enter_maintenance_log(request):
    if request.method == 'POST':
        form = MaintenanceLogForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('data_entry:enter_maintenance_log')
    else:
        form = MaintenanceLogForm()
    return render(request, 'data_entry/enter_maintenance_log.html', {'form': form})

class ProductionDataViewSet(viewsets.ModelViewSet):
    queryset = ProductionData.objects.all()
    serializer_class = ProductionDataSerializer

class EquipmentDataViewSet(viewsets.ModelViewSet):
    queryset = EquipmentData.objects.all()
    serializer_class = EquipmentDataSerializer

class MaintenanceLogViewSet(viewsets.ModelViewSet):
    queryset = MaintenanceLog.objects.all()
    serializer_class = MaintenanceLogSerializer
