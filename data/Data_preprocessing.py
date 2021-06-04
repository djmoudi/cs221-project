

import pandas as pd
import numpy as np


# --------------------Load Data Sets ---------------------------------------------------------

train_set = pd.read_csv(r'\home\cs221\project\data\train_FD001.txt', sep=" ", header=None)
test_set = pd.read_csv(r'\home\cs221\project\data\test_FD001.txt', sep=" ", header=None)
RUL_ground = pd.read_csv(r"\home\cs221\project\data\RUL_FD001.txt", sep="\s", header=None)

# Trim unnecessary columns
train_set.drop([26, 27], axis=1, inplace=True)
test_test.drop([26, 27], axis=1, inplace=True)

# Define columns
columns = ['id', 'cycle', 'st1', 'st2', 'st3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', \
             's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

features = ['st1', 'st2', 'st3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', \
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# rename columns
train_set.columns = columns
test_test.columns = columns
RUL_ground.columns = ['RUL']

# Removing low variation (standard deviation ~ 0) features to reduce input dimensions

reduced_columns = ['id', 'cycle', 'st1', 'st2', 's3', 's4', 's7', 's9', 's11', 's12', 's14', 's15', 's17', 's20', 's21']
reduced_features = ['st1', 'st2', 's3', 's4', 's7', 's9', 's11', 's12', 's14', 's15', 's17', 's20', 's21']

# -----------------------------------------------------------------------------
# Data augmentation - augmenting data set with new features to improve model performance
# Namely: moving average and rolling standard deviation

# function to calculate and add sensors moving averages and rolling standard deviations
def features_aug(df, iter):

    temp = features
    output = pd.DataFrame()

    for i in pd.unique(df.id):

        engine = df[df['id'] == i]
        features_subset = engine[temp]

        mov_average = features_subset.rolling(ri, min_periods=1).mean()
        for j in temp:
            j.replace('s', 'move_av')
        mov_average.columns = j

        sd = features_subset.rolling(iter, min_periods=1).std().fillna(0)
        for j in temp:
            j.replace('s', 'sd')
        sd.columns = j

        aug = pd.concat([engine, amove_av, sd], axis=1)
        output = pd.concat([output, aug])

    return output

# Add new features to data sets and save data in CSV files
final_train = features_aug(train_set, 9)
final_test = features_aug(test_set, 9)
final_train = final_train.to_csv(r'\home\cs221\project\data\final_train.csv', index=None, header=True, encoding='utf-8')
final_test = final_test.to_csv(r'\home\cs221\project\data\final_test.csv', index=None, header=True, encoding='utf-8')