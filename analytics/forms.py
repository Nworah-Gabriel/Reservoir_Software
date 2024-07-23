from django import forms

class AssuranceFlowForm(forms.Form):
    reservoir_pressure = forms.FloatField(label='Reservoir Pressure (psi)')
    temperature = forms.FloatField(label='Temperature (°F)')
    flow_rate = forms.FloatField(label='Flow Rate (barrels per day)')
    oil_ratio = forms.FloatField(label='Oil Ratio')
    water_ratio = forms.FloatField(label='Water Ratio')
    gas_ratio = forms.FloatField(label='Gas Ratio')
    pipeline_diameter = forms.FloatField(label='Pipeline Diameter (inches)')
    fluid_viscosity = forms.FloatField(label='Fluid Viscosity (centipoise)')
    fluid_density = forms.FloatField(label='Fluid Density (kg/m^3)')
    environment_temp = forms.FloatField(label='Environment Temperature (°C)')
    historical_production = forms.FloatField(label='Historical Production (barrels per day)')
