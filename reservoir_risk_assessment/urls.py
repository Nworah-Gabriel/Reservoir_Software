"""
URL configuration for reservoir_risk_assessment project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from data_entry import views


app_name = 'data_entry'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analytics/', include('analytics.urls')),
    path('data_entry/', include('data_entry.urls')),
    path('enter_production_data/', views.enter_production_data, name='enter_production_data'),
    path('enter_equipment_data/', views.enter_equipment_data, name='enter_equipment_data'),
    path('enter_maintenance_log/', views.enter_maintenance_log, name='enter_maintenance_log'),
]

