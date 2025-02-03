# 📈 Forecasting Stock Market Trends  

## 📌 Project Overview  
This project focuses on **predicting stock market trends** using machine learning. We analyze historical stock data from major indices (**S&P 500, NASDAQ, DJI, NYSE, Russell 2000**) to forecast closing prices. The project involves **data cleaning, exploratory analysis, feature engineering, and regression modeling** to improve prediction accuracy.  

---

## 📊 Dataset  
The dataset is sourced from the **UCI Machine Learning Repository** and contains:  
- **Closing Prices**  
- **Trading Volumes**  
- **Volatility Measures**  
- **Stock Indices (S&P 500, NASDAQ, DJI, NYSE, Russell 2000)**  

---

## 🎯 Objective  
Develop a **machine learning model** that predicts stock closing prices based on historical trends and financial indicators.  

---

## 🛠️ Data Preprocessing  
✔ **Handling Missing Values:** Interpolation, forward-fill, and backward-fill techniques.  
✔ **Outlier Detection:** Z-score method (Threshold = 3).  
✔ **Feature Scaling:** MinMaxScaler for numerical features.  
✔ **Feature Encoding:** Encoding categorical variables (e.g., stock indices).  
✔ **Date Formatting:** Datetime conversion for time-series analysis.  

---

## 📊 Exploratory Data Analysis (EDA)  
🔍 **Time Series Plots:** Visualizing stock trends.  
🔍 **Scatter Plots & Cross-Correlation:** Examining relationships between features.  
🔍 **Box Plots & Histograms:** Identifying data distributions and anomalies.  
🔍 **Heatmaps:** Understanding correlation between features.  

---

## ⚙️ Feature Engineering  
📌 **Exponential Moving Averages (EMA_10, EMA_20, EMA_200)** – Capturing short, medium, and long-term trends.  
📌 **Rate of Change (ROC) Indicators** – Measuring momentum shifts.  
📌 **Stock Index Encoding** – Converting categorical indices into numerical values.  

---

## 🤖 Machine Learning Models Used  
We implemented the following **regression models**:  
1. **Linear Regression**  
2. **K-Nearest Neighbors (KNN)**  
3. **Support Vector Machine (SVM)**  
4. **Random Forest**  
5. **Gradient Boosting** *(Best Performing Model)*  

---

## 🔧 Hyperparameter Tuning  
Performed **grid search** for optimal parameters. Best **Gradient Boosting** parameters:  
- **Learning Rate:** 0.1  
- **Number of Estimators:** 100  

---

## 📈 Model Performance Evaluation  
**Metrics Used:**  
✔ **Mean Squared Error (MSE)**  
✔ **Root Mean Squared Error (RMSE)**  
✔ **Mean Absolute Error (MAE)**  
✔ **R-Squared (R²)**  

| Model | MSE | RMSE | R² |  
|--------|------------|------------|------------|  
| **Linear Regression** | 0.012777 | 0.1129 | 0.85 |  
| **KNN** | 0.024707 | 0.1572 | 0.78 |  
| **Random Forest** | 0.000023 | 0.0048 | **0.9996** |  
| **Gradient Boosting** | **0.000018** | **0.0042** | **0.9998** |  
| **SVM** | 0.006600 | 0.0813 | 0.92 |  
| **SVR** | 0.015817 | 0.1258 | 0.88 |  

---

## 🔬 Advanced Models & Ensemble Learning  
We explored:  
✔ **XGBoost** – R²: **0.99956**  
✔ **Extreme Learning Machine (ELM)** – R²: **0.99909**  
✔ **Convolutional Neural Networks (CNNs)** – MSE: **5.0370e-05**  
✔ **Ensemble Model (XGBoost + ELM + CNNs)** – **Best Performing Model**  
   - R²: **0.99976**  
   - MSE: **2.2541e-05**  

---

## 🎯 Conclusion  
✔ **Gradient Boosting performed best among traditional ML models**.  
✔ **The ensemble model (XGBoost + ELM + CNN) achieved the highest accuracy**.  
✔ **Machine learning can effectively forecast stock market trends**.  

---

## 🚀 Future Scope  
🔹 Exploring **deep learning (LSTMs, Transformers)** for time-series forecasting.  
🔹 Implementing **real-time stock prediction pipelines**.  
🔹 Using **advanced ensemble techniques (bagging, stacking)** to boost accuracy.  

---
