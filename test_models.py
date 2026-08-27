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
    print(" TEST 2: KATHMANDU AIR QUALITY & HEALTH HAZARD CLASSIFICATION (SVC)")
    print("="*75)

    svc_model = joblib.load('02_svm_classification_nepal_air_quality/best_svc_model.joblib')

    aqi_test_cases = [
        {
            'scenario': 'Monsoon Afternoon Clean Air (July, Rain washed)',
            'Raw Conc.': 10.5, 'Month': 7, 'Hour': 14, 'Season': 'Monsoon',
            'PM25_Lag1': 12.0, 'PM25_Lag3': 15.0, 'PM25_Roll6h': 11.8
        },
        {
            'scenario': 'Spring Day Moderate Air (April Afternoon)',
            'Raw Conc.': 32.0, 'Month': 4, 'Hour': 16, 'Season': 'Spring',
            'PM25_Lag1': 30.0, 'PM25_Lag3': 28.0, 'PM25_Roll6h': 31.0
        },
        {
            'scenario': 'Winter Morning Peak Pollution (January Smog & Inversion)',
            'Raw Conc.': 125.0, 'Month': 1, 'Hour': 8, 'Season': 'Winter',
            'PM25_Lag1': 130.0, 'PM25_Lag3': 110.0, 'PM25_Roll6h': 120.0
        },
        {
            'scenario': 'Severe Winter Night Pollution Inversion (December Midnight)',
            'Raw Conc.': 210.0, 'Month': 12, 'Hour': 23, 'Season': 'Winter',
            'PM25_Lag1': 195.0, 'PM25_Lag3': 180.0, 'PM25_Roll6h': 190.0
        }
    ]

    advisories = {
        'Good': '🟢 Air quality is satisfactory. Safe for all outdoor activities.',
        'Moderate': '🟡 Acceptable. Unusually sensitive individuals should limit prolonged exertion.',
        'Unhealthy_Sensitive': '🟠 Sensitive groups (asthma, children, seniors) should reduce outdoor exertion.',
        'Unhealthy': '🔴 Everyone may experience health effects. Wear N95 mask outdoors.',
        'Hazardous': '🟣 Severe health warning! Stay indoors and keep windows closed.'
    }

    for idx, case in enumerate(aqi_test_cases, 1):
        hour = case['Hour']
        month = case['Month']
        raw_pm = case['Raw Conc.']
        lag1 = case['PM25_Lag1']
        df_in = pd.DataFrame([{
            'Raw Conc.': raw_pm,
            'NowCast Conc.': 0.6 * raw_pm + 0.4 * lag1,
            'PM25_Lag1': lag1,
            'PM25_Lag3': case['PM25_Lag3'],
            'PM25_Roll6h': case['PM25_Roll6h'],
            'Hour_Sin': np.sin(2 * np.pi * hour / 24.0),
            'Hour_Cos': np.cos(2 * np.pi * hour / 24.0),
            'Month_Sin': np.sin(2 * np.pi * month / 12.0),
            'Month_Cos': np.cos(2 * np.pi * month / 12.0),
            'Season': case['Season']
        }])
        pred_class = svc_model.predict(df_in)[0]
        
        print(f"\n[Test Case {idx}]: {case['scenario']}")
        print("  INPUT:")
        print(f"    - PM2.5 Level   : {case['Raw Conc.']} µg/m³ (1h Lag: {case['PM25_Lag1']} µg/m³, 6h Avg: {case['PM25_Roll6h']} µg/m³)")
        print(f"    - Time & Season : {case['Season']} | Month {case['Month']} | Time: {case['Hour']:02d}:00")
        print("  OUTPUT (SVC Model Prediction):")
        print(f"    -> Predicted AQI Category : {pred_class}")
        print(f"    -> Health Advisory        : {advisories.get(pred_class, '')}")

    print("\n" + "="*75)

if __name__ == "__main__":
    run_tests()
