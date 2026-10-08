# House Price Prediction

A machine learning regression project for predicting median house values
using the California Housing dataset.

## Project Overview

This project explores different regression algorithms, compares their
performance, and uses XGBoost with hyperparameter tuning to build the
final house price prediction model.

### Objectives

-   Understand and explore the California Housing dataset.
-   Handle missing and categorical data.
-   Perform exploratory data analysis (EDA).
-   Compare multiple regression algorithms using RMSE and R².
-   Tune XGBoost hyperparameters using cross-validation.
-   Analyze feature importance.
-   Evaluate actual vs predicted house values.
-   Save the trained model for future use.

## Dataset

The project uses the **California Housing dataset** with features such
as:

-   `longitude`
-   `latitude`
-   `housing_median_age`
-   `total_rooms`
-   `total_bedrooms`
-   `population`
-   `households`
-   `median_income`
-   `ocean_proximity`

### Target

-   `median_house_value`

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   XGBoost
-   Joblib
-   Jupyter Notebook

## Project Workflow

``` text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Missing Value Handling
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
XGBoost Hyperparameter Tuning
   ↓
Feature Importance Analysis
   ↓
Final Model Evaluation
   ↓
Model Saving
```

## Data Preprocessing

### Missing Values

Missing values in `total_bedrooms` were handled using mean imputation
based on the training data.

### Categorical Encoding

The `ocean_proximity` categorical feature was converted into numerical
features using `OneHotEncoder`.

The `ISLAND` category was used as the dropped reference category.

### Feature Scaling

Standardization was applied for the Linear Regression workflow.
Tree-based models such as Random Forest and XGBoost do not require
feature scaling.

## Models Evaluated

The following regression models were compared:

1.  Linear Regression
2.  Decision Tree Regressor
3.  Random Forest Regressor
4.  Gradient Boosting Regressor
5.  AdaBoost Regressor
6.  XGBoost Regressor

## Model Evaluation

Two evaluation metrics were used:

-   **RMSE (Root Mean Squared Error):** Measures the typical magnitude
    of prediction errors. Lower is better.
-   **R² (R-squared):** Measures how much variance in the target is
    explained by the model. Higher is better.

### Model Comparison

  Model                      RMSE           R²
  ------------------- ----------- ------------
  XGBoost               45,043.54   **0.8452**
  Random Forest         48,929.63       0.8173
  Gradient Boosting     55,970.26       0.7609
  Linear Regression     70,031.49       0.6257
  Decision Tree         70,162.99       0.6243

## XGBoost Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to search for better
XGBoost parameters.

The selected configuration included:

``` python
{
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_depth": 10,
    "min_child_weight": 3,
    "n_estimators": 250,
    "subsample": 0.8
}
```

### Final XGBoost Performance

``` text
Test RMSE : 45,043.54
Test R²   : 0.8452
CV R²     : 0.8446
```

The close CV and test R² scores indicate consistent performance between
cross-validation and the held-out test set.

## Feature Importance

Feature importance from the final XGBoost model showed that
`ocean_proximity_INLAND` had the highest model importance, followed by
`median_income` and other `ocean_proximity` categories.

The analysis was used to understand which features contributed most to
the model's predictions.

## Actual vs Predicted Analysis

The actual vs predicted plot showed that most predictions were
concentrated around the reference line, indicating a strong relationship
between actual and predicted house values.

Prediction errors became more noticeable for higher-valued houses, and
some observations were farther from the reference line.

## Model Saving

The trained XGBoost model was saved using Joblib:

``` python
import joblib

joblib.dump(xgb, "house_price_xgb_model.pkl")
```

The saved model can be loaded later with:

``` python
model = joblib.load("house_price_xgb_model.pkl")
```

## Project Structure

``` text
House-Price-Prediction/
│
├── House Price Prediction.ipynb
├── housing.csv
├── house_price_xgb_model.pkl
└── README.md
```

