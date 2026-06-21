# CodeAlpha_CarPricePrediction
🚗 Predicting used car resale prices using Machine Learning (Linear Regression, Random Forest &amp; Gradient Boosting) | CodeAlpha Data Science Internship Task 3
# 🚗 Car Price Prediction with Machine Learning
### CodeAlpha Data Science Internship — Task 3



![Python](https://img.shields.io/badge/Python-3.8+-blue)




![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)




![Status](https://img.shields.io/badge/Status-Completed-green)



---

## 📌 Objective
Predict the resale selling price of used cars based on features like
brand, age, mileage, fuel type, and transmission type using
supervised machine learning regression models.

---

## 📂 Dataset
| Feature | Description |
|---|---|
| Car_Name | Name/brand of the car |
| Year | Year of manufacture |
| Selling_Price | Target variable (in Lakhs ₹) |
| Present_Price | Current ex-showroom price |
| Driven_kms | Total kilometres driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Selling_type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

- **Total Records:** 173
- **Price Range:** ₹0.35L – ₹33.00L

---

## ⚙️ Feature Engineering
- Created `Car_Age` from `(2024 - Year)`
- Label encoded: `Fuel_Type`, `Selling_type`, `Transmission`
- Final features: `Present_Price`, `Driven_kms`, `Fuel_Type`,
`Selling_type`, `Transmission`, `Owner`, `Car_Age`

---

## 🤖 Models Trained

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 1.98 | 3.08 | 0.5579 |
| Random Forest | 1.23 | 1.69 | 0.8670 |
| **Gradient Boosting** ✅ | **1.06** | **1.51** | **0.8934** |

> ✅ **Best Model: Gradient Boosting** with R² = 0.8934

---

## 💡 Key Insights
- 🔹 **Present Price** is the #1 predictor of resale value
- 🔹 **Car Age** has a strong negative impact on price
- 🔹 **Diesel cars** retain value better than Petrol
- 🔹 **Automatic transmission** commands a price premium
- 🔹 **Lower mileage** positively impacts selling price
- 🔹 **Dealer-sold** cars fetch higher prices than individual sellers

---

## 🛠️ Tech Stack
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Project Structure