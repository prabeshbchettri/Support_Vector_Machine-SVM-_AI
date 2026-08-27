"""
==============================================================================
Project 2: SVC Inference Script
Kathmandu Valley Air Quality Index (AQI) Predictor & Health Advisory
==============================================================================
"""

import sys
import joblib
import numpy as np
import pandas as pd


def predict_air_quality(pm25_raw=85.0, month=1, hour=8, season="Winter", pm25_lag1=90.0, pm25_lag3=75.0, pm25_roll6h=82.0):
    """
    Predicts AQI Category in Kathmandu Valley and provides actionable health advisory.
    
    Parameters:
    - pm25_raw: Current PM2.5 reading in ug/m3
    - month: Month of year (1-12)
    - hour: Hour of day (0-23)
    - season: 'Winter', 'Spring', 'Monsoon', 'Post-Monsoon'
    - pm25_lag1: PM2.5 1 hour ago
    - pm25_lag3: PM2.5 3 hours ago
    - pm25_roll6h: 6-hour rolling average PM2.5
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
    nowcast_conc = 0.6 * pm25_raw + 0.4 * pm25_lag1
    
    sample = pd.DataFrame([{
        'Raw Conc.': pm25_raw,
        'NowCast Conc.': nowcast_conc,
        'PM25_Lag1': pm25_lag1,
        'PM25_Lag3': pm25_lag3,
        'PM25_Roll6h': pm25_roll6h,
        'Hour_Sin': hour_sin,
        'Hour_Cos': hour_cos,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'Season': season
    }])
    
    pred_category = model.predict(sample)[0]
    
    advisory = {
        'Good': '🟢 Air quality is satisfactory. Enjoy outdoor activities!',
        'Moderate': '🟡 Air quality is acceptable. Sensitive individuals should consider reducing heavy outdoor exertion.',
        'Unhealthy_Sensitive': '🟠 Sensitive groups (asthma, children, elderly) may experience health effects. Limit prolonged outdoor exertion.',
        'Unhealthy': '🔴 Everyone may begin to experience health effects. Wear N95 masks and avoid strenuous outdoor exercise.',
        'Hazardous': '🟣 Health alert: serious risk for the entire population. Stay indoors and use air purifiers.'
    }
    
    print("\n" + "="*65)
    print(" 🏭 KATHMANDU AIR QUALITY INDEX (AQI) PREDICTOR (SVC)")
    print("="*65)
    print(f" Current PM2.5 : {pm25_raw:.1f} µg/m³ (Season: {season}, Time: {hour:02d}:00, Month: {month})")
    print(f" Trend Context : 1h Lag: {pm25_lag1} µg/m³ | 6h Avg: {pm25_roll6h} µg/m³")
    print("-"*65)
    print(f" Predicted AQI Category : {pred_category.upper()}")
    print(f" Health Advisory        : {advisory.get(pred_category, '')}")
    print("="*65 + "\n")


if __name__ == "__main__":
    print("[+] Running sample Kathmandu air quality predictions...")
    # Winter morning peak pollution in Kathmandu
    predict_air_quality(pm25_raw=115.0, month=1, hour=8, season="Winter", pm25_lag1=120.0, pm25_lag3=95.0, pm25_roll6h=110.0)
    # Monsoon clean afternoon
    predict_air_quality(pm25_raw=15.0, month=7, hour=14, season="Monsoon", pm25_lag1=18.0, pm25_lag3=22.0, pm25_roll6h=16.5)
