# 📊 Retail Demand Forecasting (Department-Level)

## 🚀 Overview

This project builds a **department-level demand forecasting system** using retail data. The goal is to predict future demand by combining **econometric modeling (Log-Log OLS)** with **Bayesian regression**, while ensuring reproducibility and experiment tracking using **MLflow**.

The approach balances:

* 📈 **Interpretability** (price elasticity via log-log models)
* 🧪 **Experimentation** (MLflow tracking)
* 🔁 **Scalability** (modular pipeline design)

---

## 📂 Dataset

The dataset contains daily retail data from **01 Jan 2022 to 30 Jan 2024** with the following features:

* `Date`
* `Store ID`
* `Product ID`
* `Category` (Department)
* `Region`
* `Inventory Level`
* `Units Sold`
* `Units Ordered`
* `Price`
* `Discount`
* `Weather Condition`
* `Promotion`
* `Competitor Pricing`
* `Seasonality`
* `Epidemic`
* `Demand` (Target Variable)

---

## ⚙️ Methodology

### 1. Data Aggregation

* Aggregated at **Category (Department) + Date level**
* Enables department-level forecasting
* Weekly aggregation used for visualization

---

### 2. Feature Engineering

* Time-based features:

  * Month
  * Day of week
* Lag features:

  * 7-day lag
  * 14-day lag
* Log transformations applied to stabilize variance

---

### 3. Models

#### 🔹 Log-Log OLS Regression

* Implemented using `statsmodels`
* Provides:

  * Intercept & coefficients (elasticities)
  * Statistical significance (p-values)
* Useful for pricing and business interpretation

---

#### 🔹 Bayesian Ridge Regression

* Handles multicollinearity
* More stable under noisy retail conditions
* Provides regularized estimates

---

### 4. Train-Test Split

* **Train:** 01-01-2022 → 30-11-2023
* **Test:** 01-12-2023 → 30-01-2024

---

## 📏 Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* **WAPE (Weighted Absolute Percentage Error)** ← Primary metric

---

## 📊 Forecast Visualization

Example of weekly demand forecast vs actuals:

<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/a9bd5d6f-b0d3-4102-9d4c-c648ad662bc2" />

<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/b774bfa0-2e71-48cb-bb42-61ab5c0d98c1" />

<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/05703624-1334-4fa2-b943-bf9061245e4e" />


### 📌 To add your chart:

1. Create a folder:

   ```
   images/
   ```
2. Save your plot:

   ```
   images/forecast_example.png
   ```
3. Ensure the filename matches the one used above.

---

## 🧪 MLflow Experiment Tracking

Experiments are tracked using **MLflow** for:

* Model comparison
* Parameter tracking
* Metric logging
* Artifact storage

### Logged Information:

* Model type (OLS, Bayesian)
* Feature set used
* Evaluation metrics (WAPE, RMSE)
* Saved models
* Forecast outputs

---

## 🚀 How to Run

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Start MLflow UI

```
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

### 3. Train Models

```
python train.py
```

---

## 🔁 Scope for Improvement

### 🔹 Feature Engineering

* Rolling statistics (7-day, 30-day averages)
* Categorical encoding (promotion, weather, region)

---

### 🔹 Advanced Models

* Gradient Boosting (LightGBM, XGBoost)
* Time-series models (Prophet, SARIMAX)
* Deep learning (LSTM, DeepAR)

---

### 🔹 Hierarchical Forecasting

* Store + Category level modeling
* Region-level aggregation

---

### 🔹 Causal Modeling

* Price elasticity refinement
* Promotion uplift modeling

---

### 🔹 Productionization

* Deployment using APIs / Databricks
* Scheduled retraining pipelines
* Real-time demand forecasting

---

## 🧠 Key Learnings

* Log-log regression provides strong interpretability for demand drivers
* WAPE is more suitable than RMSE for retail forecasting
* MLflow enables structured and reproducible experimentation

---

## 📌 Future Work

* Build an interactive dashboard (Streamlit / Dash)
* Add confidence intervals to forecasts
* Integrate with real-time data pipelines

---

## 👨‍💻 Author

Debajyoti Roy
