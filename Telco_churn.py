##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem: Developing a machine learning model that can predict customers who will leave the company (churn).
# Before developing the model, it is expected to perform the necessary data analysis and feature engineering steps.

# Telco customer churn data contains information about a fictional telecom company that provided 
# home phone and Internet services to 7,043 customers in California in the third quarter. 
# It includes which customers have left, stayed, or signed up for their service.

# 21 Variables, 7043 Observations

# CustomerId : Customer unique ID
# Gender : Customer's gender
# SeniorCitizen : Whether the customer is a senior citizen (1, 0)
# Partner : Whether the customer has a partner (Yes, No) / Marital status
# Dependents : Whether the customer has dependents (Yes, No) (Child, mother, father, grandmother, etc.)
# tenure : Number of months the customer has stayed with the company
# PhoneService : Whether the customer has a phone service (Yes, No)
# MultipleLines : Whether the customer has more than one line (Yes, No, No phone service)
# InternetService : Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup : Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, No internet service)
# TechSupport : Whether the customer receives technical support (Yes, No, No internet service)
# StreamingTV : Whether the customer has streaming TV (Yes, No, No internet service)
# StreamingMovies : Whether the customer has streaming movies (Yes, No, No internet service)
# Contract : Customer's contract duration (Month-to-month, One year, Two year)
# PaperlessBilling : Whether the customer has paperless billing (Yes, No)
# PaymentMethod : Customer's payment method (Electronic check, Mailed check, Bank transfer (auto), Credit card (auto))
# MonthlyCharges : The amount charged to the customer monthly
# TotalCharges : Total amount charged to the customer
# Churn : Whether the customer churned (Yes or No) - Customers who left in the last month or quarter

# Each row represents a unique customer.
# Variables include information about customer service, account, and demographic data.
# Services: phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV/movies.
# Account info: tenure, contract, payment method, paperless billing, monthly charges, and total charges.
# Demographic info: gender, age range, and family status (partners/dependents).

# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
# Step 1: Examine the overall picture.
# Step 2: Capture numerical and categorical variables.
# Step 3: Analyze numerical and categorical variables.
# Step 4: Perform target variable analysis (Mean of target by categorical variables, mean of numerical variables by target).
# Step 5: Perform outlier analysis.
# Step 6: Perform missing value analysis.
# Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
# Step 1: Take necessary actions for missing and outlier values.
# Step 2: Create new features.
# Step 3: Perform encoding operations.
# Step 4: Standardize numerical variables.
# Step 5: Build the model.

# Required Libraries and Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load data
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# TotalCharges should be a numerical variable
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Convert target variable Churn to binary
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

##################################
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
##################################

##################################
# OVERALL PICTURE
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##################################
# CAPTURING NUMERICAL AND CATEGORICAL VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical-but-cardinal variables in the dataset.
    Note: Categorical variables also include numerical-looking categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe whose variable names are to be retrieved.
        cat_th: int, optional
                Threshold value for numerical-looking categorical variables.
        car_th: int, optional
                Threshold value for categorical-but-cardinal variables.

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                Cardinal variable list (categorical looking)
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Analyzing Tenure distribution: High frequency at 1 month and 70+ months.
# Let's check tenure distributions based on contract types.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# Analyzing MonthlyCharges: Average monthly payments might be higher for month-to-month contracts.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

##################################
# TARGET ANALYSIS BY NUMERICAL VARIABLES
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

##################################
# TARGET ANALYSIS BY CATEGORICAL VARIABLES
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##################################
# CORRELATION
##################################

# Heatmap for correlation matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Observations: TotalCharges is highly correlated with MonthlyCharges and Tenure.

##################################
# TASK 2: FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Imputing missing TotalCharges with median
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

##################################
# BASE MODEL DEVELOPMENT
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

# Evaluating multiple base models using cross-validation
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")

##################################
# OUTLIER ANALYSIS
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Check and suppress outliers
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

##################################
# FEATURE EXTRACTION
##################################

# Creating categorical Year variable from Tenure
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Define customers with 1 or 2 year contracts as "Engaged"
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Identify customers without protection/backup/support services
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Young customers with month-to-month contracts
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Total number of services used by the customer
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Flag for any streaming services used
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Automatic payment flag
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Average monthly charges calculated from total charges
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Price increase ratio
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Average fee per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

##################################
# ENCODING
##################################

# Re-identifying variable types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding for binary columns
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding for remaining categorical variables
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]

df = one_hot_encoder(df, cat_cols, drop_first=True)

##################################
# MODELLING
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

# Compare final models with engineered features
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")

################################################
# HYPERPARAMETER OPTIMIZATION (RF, XGBoost, LightGBM, CatBoost)
################################################

# Feature Importance visualization
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

# plot_importance(rf_final, X)