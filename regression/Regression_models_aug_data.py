

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


# Define and load augmented data set features
features = ['st1', 'st2','st3','s1',  's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9','s10',\
            's11', 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','avt1','avt2','avt3',\
            'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9','av10','av11','av12','av13',\
            'av14','av15','av16','av17','av18','av19','av20','av21','sdt1','sdt2','sdt3',\
            'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9','sd10','sd11','sd12','sd13',\
            'sd14','sd15','sd16','sd17','sd18','sd19','sd20','sd21']

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
