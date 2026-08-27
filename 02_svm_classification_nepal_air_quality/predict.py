"""
==============================================================================
Project 2: SVC Inference Script
Kathmandu Valley Air Quality & Atmospheric Risk Predictor
Dataset: Kathmandu AQI Dataset (2022-2025) by Subesh Yadav
==============================================================================
"""

import sys
import joblib
import numpy as np
import pandas as pd


def predict_air_quality_risk(temp_c=10.5, humidity_pct=88.0, apparent_temp_c=9.2,
                             wind_10m=2.5, wind_100m=3.8, soil_surf=0.38,
                             soil_deep=0.39, hour=8, month=1, season="Winter"):
    """
    Predicts Kathmandu Valley Atmospheric Pollution Risk using the trained SVC model.
    
    Parameters:
    - temp_c: Temperature in °C (2m)
    - humidity_pct: Relative humidity in % (2m)
    - apparent_temp_c: Perceived / apparent temperature in °C
    - wind_10m: Wind speed at 10m altitude in km/h
    - wind_100m: Wind speed at 100m altitude in km/h
    - soil_surf: Surface soil moisture (0-7cm)
    - soil_deep: Deep soil moisture (7-28cm)
    - hour: Hour of day (0-23)
    - month: Month of year (1-12)
    - season: 'Winter', 'Spring', 'Monsoon', 'Post-Monsoon'
    """
    model_path = "best_svc_model.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Please run train_svc.py first.")
        return
        
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    wind_shear = wind_100m - wind_10m
    
    sample = pd.DataFrame([{
        'Temperature_C': temp_c,
        'Humidity_Pct': humidity_pct,
        'Apparent_Temp_C': apparent_temp_c,
        'Wind_Speed_10m': wind_10m,
        'Wind_Speed_100m': wind_100m,
        'Wind_Shear': wind_shear,
        'Soil_Moisture_Surface': soil_surf,
        'Soil_Moisture_Deep': soil_deep,
        'Hour_Sin': hour_sin,
        'Hour_Cos': hour_cos,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'Season': season
    }])
    
    pred_risk = model.predict(sample)[0]
    
    advisory = {
        'Hazardous_Inversion': '🟣 CRITICAL ALERT: Strong thermal inversion trapping dense winter smog. Avoid outdoor exercise; use N95 masks indoors/outdoors.',
        'High_Stagnation': '🔴 HIGH SMOG RISK: Low surface ventilation. Sensitive groups should stay indoors.',
        'Moderate_Dispersion': '🟡 MODERATE DISPERSION: Typical valley ventilation. Acceptable for general public.',
        'Good_Ventilation': '🟢 ACTIVE VENTILATION: Strong winds clearing particulates. Optimal air dispersion.'
    }
    
    print("\n" + "="*70)
    print(" 🏭 KATHMANDU AIR QUALITY & INVERSION RISK PREDICTOR (SVC)")
    print("="*70)
    print(f" Atmospheric State : Temp: {temp_c}°C | Humidity: {humidity_pct}% | Wind (10m): {wind_10m} km/h")
    print(f" Time & Context    : {season} | Month {month} | Time: {hour:02d}:00")
    print("-"*70)
    print(f" Predicted Risk Level : {pred_risk}")
    print(f" Health & City Action : {advisory.get(pred_risk, '')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("[+] Running sample predictions on Kathmandu 2022-2025 meteorological data...")
    # Test 1: Cold winter morning smog inversion
    predict_air_quality_risk(temp_c=8.5, humidity_pct=92.0, apparent_temp_c=7.0, wind_10m=1.8, wind_100m=2.5, hour=8, month=1, season="Winter")
    # Test 2: Monsoon ventilated afternoon
    predict_air_quality_risk(temp_c=26.5, humidity_pct=65.0, apparent_temp_c=30.0, wind_10m=14.0, wind_100m=18.5, hour=14, month=7, season="Monsoon")
