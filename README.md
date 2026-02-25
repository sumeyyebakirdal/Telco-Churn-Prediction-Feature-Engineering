# Telco Customer Churn: Feature Engineering & Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-Churn_Prediction-red)

## 📌 Business Problem
Customer churn (loss) is one of the most critical challenges for telecommunications companies. This project aims to develop a machine learning model that predicts which customers are likely to leave the company based on their demographic information, account details, and service usage patterns.

The goal is to provide actionable insights for the marketing and retention departments to proactively engage with high-risk customers.

---

## 📂 Dataset Information
The dataset used in this project is the **Telco Customer Churn** dataset.

* **Dataset Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Total Observations:** 7,043 customers
* **Total Variables:** 21 (Demographics, Services, and Account Information)
* **Target Variable:** `Churn` (Yes/No)

---

## 🛠️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
* **Variable Identification:** Automated detection of categorical, numerical, and cardinal variables.
* **Target Analysis:** Examining churn rates across different contract types and payment methods.
* **Correlation:** Analyzing relationships between `Tenure`, `MonthlyCharges`, and `TotalCharges`.

### 2. Feature Engineering (The Core)
This project emphasizes the power of feature creation to improve model performance. New features include:
* **NEW_TENURE_YEAR:** Converting months to yearly categories.
* **NEW_Engaged:** Identifying customers with long-term (1-2 year) contracts.
* **NEW_TotalServices:** Calculating the total number of services each customer uses.
* **NEW_AVG_Service_Fee:** Determining the average cost per service.
* **NEW_Increase:** Analyzing the ratio of the latest charges to average charges.



### 3. Data Preprocessing
* **Missing Values:** Handled `TotalCharges` missing values using median imputation.
* **Outliers:** Identified and suppressed outliers using the Interquartile Range (IQR) method.
* **Encoding:** Applied **Label Encoding** for binary features and **One-Hot Encoding** for multi-class categorical features.

### 4. Machine Learning Modeling
We benchmarked several algorithms using 10-fold cross-validation:
* Logistic Regression / KNN / CART
* Random Forest / SVM
* **XGBoost / LightGBM**
* **CatBoost** (Best performing model after hyperparameter optimization)

---

## 📊 Key Results & Insights
* **Contract Type:** Customers with "Month-to-month" contracts have a significantly higher churn rate.
* **Service Quality:** Fiber Optic users show higher churn rates, suggesting potential service satisfaction issues.
* **Tenure:** The first 12 months are the most critical; churn probability decreases significantly as loyalty grows.



---

## 🚀 Installation & Usage
1.  Clone the repository:
    ```bash
    git clone [https://github.com/sumeyyebakirdal/Telco-Churn-Feature-Engineering.git](https://github.com/sumeyyebakirdal/Telco-Churn-Feature-Engineering.git)
    ```
2.  Download the data from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as `Telco-Customer-Churn.csv` in the project folder.
3.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
    ```
4.  Run the analysis script or notebook.

---
