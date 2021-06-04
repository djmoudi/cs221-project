
# Load modules
import pandas as pd
import numpy as np
from sklearn import metrics

# regression performance metrics
def display_regression_metrics(model, truth, infer):

    metrics = {'RMSE': metrics.mean_squared_error(truth, infer)**0.5,}
    metrics_frame = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_frame.columns = [model]
    return metrics_frame

# function to add regression and classification labels to training data set.

def prep_training_data(train_frame, itr):


    df_max_cycle = pd.DataFrame(train_frame.groupby('id')['cycle'].max())
    df_max_cycle.reset_index(level=0, inplace=True)
    df_max_cycle.columns = ['id', 'last_cycle']

    # add time-to-failure RUL for regression models inference
    train_frame = pd.merge(train_frame, df_max_cycle, on='id')
    train_frame['RUL'] = train_frame['last_cycle'] - train_frame['cycle']
    train_frame.drop(['last_cycle'], axis=1, inplace=True)

    # add binary classes for classification models inference
    train_frame['bclass'] = train_frame['RUL'].apply(lambda x: 1 if x <= itr else 0)

    return train_frame


# function to add regression and classification labels to test data set.

def prep_test_data(test_frame, truth, itr):

    # add time-to-failure RUL for regression models inference
    last_cycle = pd.DataFrame(test_frame.groupby('id')['cycle'].max())
    last_cycle.reset_index(level=0, inplace=True)
    last_cycle.columns = ['id', 'last_cycle']

    test_frame = pd.merge(test_frame, last_cycle, on='id')
    test_frame = test_frame[test_frame['cycle'] == test_frame['last_cycle']]
    test_frame.drop(['last_cycle'], axis=1, inplace=True)

    test_frame.reset_index(drop=True, inplace=True)
    test_frame = pd.concat([test_frame, truth], axis=1)

    # add binary classes for classification models inference
    test_frame['bclass'] = test_frame['RUL'].apply(lambda x: 1 if x <= itr else 0)

    return test_frame

# Prepare the data:

# load data
train_data = pd.read_csv(r'\home\cs221\project\data\final_train.csv')
test_data  = pd.read_csv(r'\home\cs221\project\data\final_test.csv')


# add training labels
train_w_labels = prep_training_data(train_data, 20)

# add test labels
test_w_labels = prep_test_data(test_data, TrueRUL, 20)

# save CSV files
train_w_labels = train_w_labels.to_csv(r'\home\cs221\project\data\final_train_w_labels.csv',index = None, header = True, encoding='utf-8')
test_w_labels = test_w_labels.to_csv(r'\home\cs221\project\data\final_test_w_labels.csv',index = None, header = True, encoding='utf-8')

