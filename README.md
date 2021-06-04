# cs221-project
cs221 project

folder structure:

Classification: contains classification models and data prepapration Python files.

Data_prep.py: Python code to prepare data set and labels for classification models

Classification_models_original_data.py: Python code to run baseline and main approach models on original features data set

Classification_models_aug_data.py: Python code to run baseline and main approach models on augmented features data set

Classification_models_reduced_data.py: Python code to run baseline and main approach models on reduced features data set


Regression: contains regression models and data prepapration Python files.

Data_prep.py: Python code to prepare data set and labels for regression models

Regression_models_original_data.py: Python code to run baseline and main approach models on original features data set

Regression_models_aug_data.py: Python code to run baseline and main approach models on augmented features data set

Regression_models_reduced_data.py: Python code to run baseline and main approach models on reduced features data set


Data: contains raw data set files and data preprocessing Pythin file

Data_preprocessing.py: Python code to preprocess data set
train_FD001.txt: training data set
test_FD001.txt: test data set
RUL_FD001.txt: ground truth data

Program running sequence:

1- download and save the data files localy
2- Run Data_preprocessing.py to pre-process data set
3 - Regression
	Run Data_prep.py to o prepare data set and labels for regression models
	Run Regression_Models_* python files for each data set
4 - Classification
	Run Data_prep.py to o prepare data set and labels for classification models
	Run Classification_Models_* python files for each data set

