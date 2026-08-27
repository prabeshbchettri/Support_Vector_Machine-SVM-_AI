"""
==============================================================================
Project 1: SVR Inference Script
Kathmandu Valley House Price Estimator (in NPR / Lakhs / Crores)
==============================================================================
"""

import sys
import joblib
import numpy as np
import pandas as pd


def predict_house_price(city="Kathmandu", area_aana=4.0, bedrooms=4, bathrooms=3, 
                        floors=2.5, parking=1, road_width_ft=14.0, face="East"):
    """
    Predicts house price in Kathmandu Valley using the trained SVR model.
    
    Parameters:
    - city: e.g. 'Kathmandu', 'Lalitpur', 'Bhaktapur', 'Pokhara'
    - area_aana: Land area in Aana (1 Aana = 342.25 sq ft)
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - floors: Number of floors (e.g., 2.5)
    - parking: Number of parking spaces (cars)
    - road_width_ft: Road width access in Feet (e.g., 14 ft)
    - face: Cardinal face ('East', 'West', 'North', 'South', etc.)
    """
    model_path = "best_svr_model.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Please run train_svr.py first.")
        return
        
    area_sqft = area_aana * 342.25
    
    input_data = pd.DataFrame([{
        'Area_SqFt': area_sqft,
        'Bedroom': bedrooms,
        'Bathroom': bathrooms,
        'Floors': floors,
        'Parking': parking,
        'Road_Width_Ft': road_width_ft,
        'City': city,
        'Face': face
    }])
    
    # Predict log price and invert
    pred_log = model.predict(input_data)[0]
    pred_lakhs = np.expm1(pred_log)
    pred_npr = pred_lakhs * 100_000.0
    pred_crores = pred_lakhs / 100.0
    
    print("\n" + "="*55)
    print(" 🏡 KATHMANDU HOUSE PRICE ESTIMATOR (SVR)")
    print("="*55)
    print(f" Location       : {city}")
    print(f" Land Area      : {area_aana} Aana ({area_sqft:.1f} Sq. Ft.)")
    print(f" Layout         : {bedrooms} Bed, {bathrooms} Bath, {floors} Floors")
    print(f" Parking/Access : {parking} Cars | {road_width_ft} ft Road | Facing {face}")
    print("-"*55)
    print(f" Estimated Price: NPR {pred_npr:,.0f}")
    print(f"                : {pred_lakhs:.2f} Lakhs ({pred_crores:.2f} Crores NPR)")
    print("="*55 + "\n")


if __name__ == "__main__":
    # Example test predictions
    print("[+] Running sample house price estimates in Nepal...")
    predict_house_price(city="Kathmandu", area_aana=4.0, bedrooms=5, bathrooms=4, floors=2.5, parking=2, road_width_ft=16.0, face="East")
    predict_house_price(city="Lalitpur", area_aana=3.5, bedrooms=4, bathrooms=3, floors=2.0, parking=1, road_width_ft=13.0, face="South")
