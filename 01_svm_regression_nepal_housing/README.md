# Project 1: Support Vector Regression (SVR) — Kathmandu Valley House Price Prediction

## 📌 Project Overview
This project uses **Support Vector Regression (SVR)** to predict residential house prices in Nepal (Kathmandu, Lalitpur, Bhaktapur, Pokhara, Chitwan) based on real property listings.

## 📂 Dataset Information
- **Source**: Real estate listings from Nepal (Ghar Jagga).
- **Features**:
  - `Area_SqFt`: Land area converted to Square Feet (from Nepali units: Aana, Ropani, Haat).
  - `Bedroom`: Number of bedrooms.
  - `Bathroom`: Number of bathrooms.
  - `Floors`: Number of floors (e.g., 2.5, 3.0).
  - `Parking`: Vehicle parking spaces.
  - `Road_Width_Ft`: Access road width in feet.
  - `City`: Municipality/City (Kathmandu, Lalitpur, Bhaktapur, etc.).
  - `Face`: Cardinal direction orientation (East, West, North, South).
- **Target**: `Price_Lakhs` (in Lakhs NPR, where 1 Lakh = 100,000 NPR).

## 🧠 SVR Theory & Mathematical Model
Support Vector Regression maps input features into a high-dimensional feature space using kernel functions ($K(x, x')$) and builds an optimal hyperplane with an $\varepsilon$-insensitive tube:

$$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$
$$\text{subject to } \begin{cases} y_i - (\langle w, \phi(x_i) \rangle + b) \le \varepsilon + \xi_i \\ (\langle w, \phi(x_i) \rangle + b) - y_i \le \varepsilon + \xi_i^* \\ \xi_i, \xi_i^* \ge 0 \end{cases}$$

- **$\varepsilon$ (Epsilon)**: Defines the margin of tolerance where no penalty is given to errors.
- **$C$ (Regularization)**: Controls the trade-off between model flatness and error tolerance.
- **$\gamma$ (Gamma / Kernel Coefficient)**: Defines the influence radius of individual support vectors in the RBF kernel.

## 🚀 How to Run

### 1. Train Model & Generate Evaluation Plots
```bash
python train_svr.py
```

### 2. Predict House Price for Custom Input
```bash
python predict.py
```

### 3. Run Interactive Jupyter Notebook
```bash
jupyter notebook nepal_housing_svr.ipynb
```

## 📊 Outputs
- `plots/actual_vs_predicted.png`: Actual vs. SVR Predicted Prices scatter with ideal fit line.
- `plots/residuals.png`: Distribution of prediction error residuals.
- `plots/kernel_comparison.png`: Comparison of Linear, RBF, and Poly kernels.
- `best_svr_model.joblib`: Serialized trained pipeline.
