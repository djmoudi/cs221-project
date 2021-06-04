

# Load modules

import numpy as np
import pandas as pd
import Data_prep
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection



# load complete data set
train_set = pd.read_csv(r'\home\cs221\project\data\final_train_w_labels.csv')
test_set = pd.read_csv(r'\home\cs221\project\data\final_test_w_labels.csv')


# Define and load reduced data set features
features = ['st1','st2', 's3', 's4','s7','s9', 's11','s12','s14','s15','s17','s20','s21']

X_train = train_set[features]
y_train = train_set['bclass']

X_test = test_set[features]
y_test = test_set['bclass']

# Baseline logistic regression model

model = 'Logistic Regression'
# parameters for grid-searching
clf = LogisticRegression(random_state = 49)
params = {'C': [0.01, 0.1, 1.0, 10]}

grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=6, scoring='recall', n_jobs=-1)

grid_search.fit(X_train, y_train)
infer = grid_search.predict(X_test)

if hasattr(grid_search, 'predict_proba'):
    score = grid_search.predict_proba(X_test)[:, 1]
elif hasattr(grid_search, 'decision_function'):
    score = grid_search.decision_function(X_test)
else:
    score = infer

predictions = {'inference': infer, 'score': score}
predictions_frame = pd.DataFrame.from_dict(predictions)

# display perf metrics
display_classification_metrics(model, y_test, predictions_frame.inference, predictions_frame.score)

# ---------------------------------------------------------------------------------------------------
# Main classification approach  - Gaussian Naive Bayesian
model = 'Gaussian Naive Bayesian'

# parameters for grid-searching
clf = GaussianNB()
params = {}

grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=6, scoring='recall', n_jobs=-1)

grid_search.fit(X_train, y_train)
infer = grid_search.predict(X_test)

if hasattr(grid_search, 'predict_proba'):
    score = grid_search.predict_proba(X_test)[:, 1]
elif hasattr(grid_search, 'decision_function'):
    score = grid_search.decision_function(X_test)
else:
    score = infer

predictions = {'inference': infer, 'score': score}
predictions_frame = pd.DataFrame.from_dict(predictions)

# display perf metrics
display_classification_metrics(model, y_test, predictions_frame.inference, predictions_frame.score)

