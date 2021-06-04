

# Load modules
import scikitplot as skplt
import numpy as np
import pandas as pd
import Data_prep
from sklearn import metrics
from sklearn import linear_model
from xgboost import XGBRegressor



# load complete data set
train_set = pd.read_csv(r'\home\cs221\project\data\final_train_w_labels.csv')
test_set = pd.read_csv(r'\home\cs221\project\data\final_test_w_labels.csv')


# Define and load reduced data set features
features = ['st1','st2','s3','s4','s7','s9','s11','s12','s14','s15','s17','s20','s21']

X_train = train_set[features]
y_train = train_set['RUL']

X_test = test_set[features]
y_test = test_set['RUL']

# Baseline regression LASSO model

lasso = linear_model.Lasso(alpha=0.0005)
lasso.fit(X_train, y_train)
train_infer = lasso.predict(X_train)
test_infer = lasso.predict(X_test)


# display perf metrics
display_regression_metrics('LASSO', y_test, train_infer)
display_regression_metrics('LASSO', y_test, test_infer)


# Main regression approach - XGBoost
xgb = XGBRegressor(max_depth=9, learning_rate=0.001, reg_alpha=1, reg_lambda=0)

xgb.fit(X_train, y_train)
train_infer = xgb.predict(X_train)
test_infer = xgb.predict(X_test)

# display perf metrics
display_regression_metrics('XGBoost', y_test, train_infer)
display_regression_metrics('XGBoost', y_test, test_infer)
