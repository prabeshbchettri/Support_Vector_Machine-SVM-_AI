# Project 2: Support Vector Classification (SVC) — Kathmandu Valley Air Quality (2022 - 2025)

## 📌 Project Overview
This project uses **Support Vector Classification (SVC)** to classify Kathmandu Valley air quality and atmospheric pollution inversion risk into 4 environmental hazard tiers based on continuous hourly meteorological & air data from 2022 to 2025.

- **Dataset Source**: [Kathmandu AQI Dataset 2022-2025](https://www.kaggle.com/datasets/subeshyadav/kathmandu-aqi-dataset-2022-2025) by **Subesh Yadav** on Kaggle.

## 📂 Dataset Features
- `Temperature_C`: Air temperature at 2m ($^\circ\text{C}$).
- `Humidity_Pct`: Relative humidity at 2m ($\%$) - Key driver for hygroscopic smog growth.
- `Apparent_Temp_C`: Perceived / heat index temperature.
- `Wind_Speed_10m`: Surface wind velocity ($\text{km/h}$) - Controls valley horizontal ventilation.
- `Wind_Speed_100m`: Upper boundary wind speed ($\text{km/h}$).
- `Wind_Shear`: Vertical wind difference ($\text{Wind}_{100m} - \text{Wind}_{10m}$) indicating atmospheric mixing.
- `Soil_Moisture_Surface` & `Soil_Moisture_Deep`: Ground moisture index.
- `Hour_Sin`, `Hour_Cos`, `Month_Sin`, `Month_Cos`: Cyclical diurnal and annual time encodings.
- `Season`: Winter, Spring, Monsoon, Post-Monsoon.

## 🎯 Target Risk Levels (`AQI_Risk_Level`)
1. `Hazardous_Inversion`: Cold morning winter inversion with high humidity ($>75\%$) and calm winds ($\le 4\text{ km/h}$) trapping dense smog.
2. `High_Stagnation`: Calm surface air ($\le 6\text{ km/h}$) with poor particle dispersal.
3. `Moderate_Dispersion`: Typical valley background ventilation ($6 - 12\text{ km/h}$).
4. `Good_Ventilation`: Active dispersion ($\ge 12\text{ km/h}$ or strong convective mixing).

## 🚀 How to Run

### 1. Train Model & Generate Evaluation Plots
```bash
python train_svc.py
```

### 2. Predict Risk for Custom Atmospheric Readings
```bash
python predict.py
```

### 3. Run Interactive Jupyter Notebook
```bash
jupyter notebook kathmandu_air_quality_svc.ipynb
```

## 📊 Outputs
- `plots/confusion_matrix.png`: Multi-class confusion matrix heatmap.
- `plots/decision_boundary_2d.png`: PCA 2D non-linear decision boundary & support vector map.
- `plots/kernel_comparison.png`: Accuracy & Macro F1-score comparison (Linear vs RBF vs Poly).
- `best_svc_model.joblib`: Serialized trained pipeline.
