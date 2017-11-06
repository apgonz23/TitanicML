'''
Titanic Machine Learning Project
from Kaggle
'''

import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

#classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from feature_setup import setup_data


def main():
	#reads training data 
	train_data = pd.read_csv("train.csv") 
	
	#reads and joins the two testing data files
	test_data = pd.read_csv("test.csv")
	test_labels = pd.read_csv("gender_submission.csv")
	new_test_data = pd.concat([test_labels, test_data], axis=1)


	# adjusts features to be used in classifier
	train_data = setup_data(train_data)
	new_test_data = setup_data(new_test_data)

	#sets features and labels to the appropriate data
	labels_train = train_data['Survived']
	features_train = train_data.drop(['Survived'], axis=1)
	labels_test = new_test_data['Survived']
	features_test = new_test_data.drop(['Survived'], axis=1)

	#print(features_test)

	#CLASSIFIERS
	svc = SVC(C=1.0, kernel='linear', gamma='auto') # 98%
	naive_bayes = GaussianNB() # 94%
	decision_tree = DecisionTreeClassifier(min_impurity_decrease=0.09) # 94%
	forest = RandomForestClassifier(min_impurity_decrease=0.10) # 100%
	knn = KNeighborsClassifier()
	logic_reg = LogisticRegression()

	#print(features_test)
	#print(labels_train)

	#fitting classifiers
	'''
	accuracy_dict = {} 
	clf_list = [('svc',svc), ('naive_bayes',naive_bayes), ('decsion_tree',decision_tree), ('random_forest', forest), ('knn',knn), ('logistic_regression',logic_reg)]
	
	for name,clf in clf_list:
		print('Clf:', name, end=' ')
		accuracy_dict[name] = test_classifier(clf, features_train, labels_train, features_test, labels_test)
	#print(accuracy_dict)
	'''

	# BEST CLASSIFIER: RandomForest ####################
	print('Random Forest Classifier', end=' ')
	best_accuracy = test_classifier(forest, features_train, labels_train, features_test, labels_test)


# used to faclitate classifier testing
def test_classifier(clf, features_train, labels_train, features_test, labels_test):
	clf.fit(features_train, labels_train)
	predictions = clf.predict(features_test)
	accuracy = accuracy_score(predictions, labels_test)
	print("accuracy:", accuracy)
	return accuracy


#####################################
#Main function call

if __name__ == '__main__':
	main()

#####################################