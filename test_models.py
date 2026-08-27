"""
==============================================================================
Test Suite for Both SVM Projects (Regression & Classification)
Demonstrates live inputs and model predictions for Nepal real-world use cases.
==============================================================================
"""

import joblib
import numpy as np
import pandas as pd

def run_tests():
    print("="*75)
    print(" TEST 1: KATHMANDU VALLEY HOUSE PRICE PREDICTION (SVR REGRESSION)")
    print("="*75)

    svr_model = joblib.load('01_svm_regression_nepal_housing/best_svr_model.joblib')

    housing_test_cases = [
        {
            'description': 'Typical 4-Aana Family House in Kathmandu (Baneshwor/Budhanilkantha)',
            'City': 'Kathmandu', 'area_aana': 4.0, 'Bedroom': 5, 'Bathroom': 4,
            'Floors': 2.5, 'Parking': 2, 'Road_Width_Ft': 16.0, 'Face': 'East'
        },
        {
            'description': 'Compact 3-Aana Townhouse in Lalitpur (Jhamsikhel/Imadol)',
            'City': 'Lalitpur', 'area_aana': 3.0, 'Bedroom': 3, 'Bathroom': 2,
            'Floors': 2.0, 'Parking': 1, 'Road_Width_Ft': 12.0, 'Face': 'South'
        },
        {
            'description': 'Spacious 8-Aana Villa in Bhaktapur (Sallaghari)',
            'City': 'Bhaktapur', 'area_aana': 8.0, 'Bedroom': 6, 'Bathroom': 5,
            'Floors': 3.0, 'Parking': 3, 'Road_Width_Ft': 20.0, 'Face': 'North East'
        }
    ]

    for idx, case in enumerate(housing_test_cases, 1):
        area_sqft = case['area_aana'] * 342.25
        df_in = pd.DataFrame([{
            'Area_SqFt': area_sqft,
            'Bedroom': case['Bedroom'],
            'Bathroom': case['Bathroom'],
            'Floors': case['Floors'],
            'Parking': case['Parking'],
            'Road_Width_Ft': case['Road_Width_Ft'],
            'City': case['City'],
            'Face': case['Face']
        }])
        pred_log = svr_model.predict(df_in)[0]
        pred_lakhs = np.expm1(pred_log)
        pred_npr = pred_lakhs * 100_000.0
        
        print(f"\n[Test Case {idx}]: {case['description']}")
        print("  INPUT:")
        print(f"    - City          : {case['City']}")
        print(f"    - Land Area     : {case['area_aana']} Aana ({area_sqft:.1f} sq ft)")
        print(f"    - Bedrooms/Baths: {case['Bedroom']} Bed / {case['Bathroom']} Bath / {case['Floors']} Floors")
        print(f"    - Parking/Road  : {case['Parking']} Car(s) / {case['Road_Width_Ft']} ft Road / Facing {case['Face']}")
        print("  OUTPUT (SVR Model Prediction):")
        print(f"    -> Predicted Price : NPR {pred_npr:,.0f} ({pred_lakhs:.2f} Lakhs / {pred_lakhs/100:.2f} Crores)")

    print("\n" + "="*75)
    print(" TEST 2: KATHMANDU AIR QUALITY & INVERSION RISK (SVC CLASSIFICATION)")
    print(" Dataset: Kathmandu AQI Dataset (2022-2025) by Subesh Yadav")
    print("="*75)

    svc_model = joblib.load('02_svm_classification_nepal_air_quality/best_svc_model.joblib')

    aqi_test_cases = [
        {
            'scenario': 'Severe Winter Morning Smog Inversion (January 8:00 AM)',
            'temp_c': 8.5, 'humidity_pct': 92.0, 'apparent_temp_c': 7.0,
            'wind_10m': 1.8, 'wind_100m': 2.5, 'soil_surf': 0.38, 'soil_deep': 0.39,
            'hour': 8, 'month': 1, 'season': 'Winter'
        },
        {
            'scenario': 'High Stagnation Calm Autumn Day (November Evening)',
            'temp_c': 17.0, 'humidity_pct': 68.0, 'apparent_temp_c': 17.0,
            'wind_10m': 4.5, 'wind_100m': 6.0, 'soil_surf': 0.30, 'soil_deep': 0.32,
            'hour': 19, 'month': 11, 'season': 'Post-Monsoon'
        },
        {
            'scenario': 'Active Monsoon Convective Ventilation (July Afternoon)',
            'temp_c': 26.5, 'humidity_pct': 65.0, 'apparent_temp_c': 30.0,
            'wind_10m': 14.0, 'wind_100m': 18.5, 'soil_surf': 0.42, 'soil_deep': 0.43,
            'hour': 14, 'month': 7, 'season': 'Monsoon'
        }
    ]

    advisories = {
        'Hazardous_Inversion': '🟣 CRITICAL ALERT: Strong thermal inversion trapping dense winter smog. Avoid outdoor exercise; use N95 masks.',
        'High_Stagnation': '🔴 HIGH SMOG RISK: Low surface ventilation. Sensitive groups should stay indoors.',
        'Moderate_Dispersion': '🟡 MODERATE DISPERSION: Typical valley ventilation. Acceptable for general public.',
        'Good_Ventilation': '🟢 ACTIVE VENTILATION: Strong winds clearing particulates. Optimal air dispersion.'
    }

    for idx, case in enumerate(aqi_test_cases, 1):
        hour = case['hour']
        month = case['month']
        w10 = case['wind_10m']
        w100 = case['wind_100m']
        df_in = pd.DataFrame([{
            'Temperature_C': case['temp_c'],
            'Humidity_Pct': case['humidity_pct'],
            'Apparent_Temp_C': case['apparent_temp_c'],
            'Wind_Speed_10m': w10,
            'Wind_Speed_100m': w100,
            'Wind_Shear': w100 - w10,
            'Soil_Moisture_Surface': case['soil_surf'],
            'Soil_Moisture_Deep': case['soil_deep'],
            'Hour_Sin': np.sin(2 * np.pi * hour / 24.0),
            'Hour_Cos': np.cos(2 * np.pi * hour / 24.0),
            'Month_Sin': np.sin(2 * np.pi * month / 12.0),
            'Month_Cos': np.cos(2 * np.pi * month / 12.0),
            'Season': case['season']
        }])
        pred_class = svc_model.predict(df_in)[0]
        
        print(f"\n[Test Case {idx}]: {case['scenario']}")
        print("  INPUT:")
        print(f"    - Atmospheric State: {case['temp_c']}°C | {case['humidity_pct']}% RH | Wind: {w10} km/h (10m) / {w100} km/h (100m)")
        print(f"    - Time & Season    : {case['season']} | Month {month} | Time: {hour:02d}:00")
        print("  OUTPUT (SVC Model Prediction):")
        print(f"    -> Predicted Risk Tier : {pred_class}")
        print(f"    -> Action & Advisory   : {advisories.get(pred_class, '')}")

    print("\n" + "="*75)

if __name__ == "__main__":
    run_tests()
