'''
Titanic Machine Learning Project
from Kaggle
'''

import pandas as pd

training_data = pd.read_csv("train.csv") 

print(training_data.head())

training_data.isnull().sum()
