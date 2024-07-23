from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('predict_assurance/', views.predict_assurance_flow, name='predict_assurance_flow'),
    path('predict_maintenance/', views.predict_maintenance_cost, name='predict_maintenance'),
#     path('predict_flow_assurance/', views.predict_flow_assurance, name='predict_flow_assurance'),
]
