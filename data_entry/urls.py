from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'production_data', views.ProductionDataViewSet)
router.register(r'equipment_data', views.EquipmentDataViewSet)
router.register(r'maintenance_logs', views.MaintenanceLogViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
