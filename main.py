'''
Titanic Machine Learning Project
from Kaggle
'''

import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv

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

##########################################################################################################
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

	# Creates training and testing data from the original training data 
	#sub_features_train, sub_features_test, sub_labels_train, sub_labels_test = split_data_practice(train_data)

	#Sets training and testing data to the entire original training and testing data
	features_train, features_test, labels_train = split_data(train_data, new_test_data)

	#CLASSIFIERS
	svc = SVC(C=1.0,kernel='rbf',gamma='auto',class_weight='balanced',probability=True) # 77% on Kaggle
	naive_bayes = GaussianNB() # 78% on Kaggle
	decision_tree = DecisionTreeClassifier(min_samples_split=10)
	forest = RandomForestClassifier(n_estimators=20, min_samples_split=10)
	knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='ball_tree') 
	logic_reg = LogisticRegression(C=10.0)

	#Testing all classifiers with practice data
	accuracy_list = [] 
	clf_list = [('SVC',svc), ('Naive_Bayes',naive_bayes), ('Decsion_Tree',decision_tree), ('Random_Forest', forest), ('KNN',knn), ('Logistic_Regression',logic_reg)]
	'''
	for name,clf in clf_list:
		print(name, end=' ')
		accuracy = test_classifier(clf, sub_features_train, sub_labels_train, sub_features_test, sub_labels_test)
		accuracy_list.append((name, accuracy))

	#Record accuracies
	write_accuracies(accuracy_list)
	'''
	#Testing Individual Classifier with actual data
	forest = train_classifier(forest, features_train, labels_train)
	predictions = make_predictions(forest, features_test)
	write_submissions(predictions)
##########################################################################################################


#splits data into actual training and testing sets
def split_data(train_data, test_data):
	labels_train = train_data['Survived']
	features_train = train_data.drop(['Survived'], axis=1)
	features_test = test_data.drop(['Survived'], axis=1)
	return (features_train, features_test, labels_train)	


#splits data into a practice training and testing feature/label set
def split_data_practice(train_data):
	sub_train, sub_test = train_test_split(train_data, test_size=0.3, random_state=0, stratify=train_data['Survived'])
	features_train = sub_train[sub_train.columns[1:]]
	labels_train =  sub_train[sub_train.columns[:1]]['Survived']
	features_test = sub_test[sub_test.columns[1:]]
	labels_test = sub_test[sub_test.columns[:1]]
	return (features_train, features_test, labels_train, labels_test)


def write_accuracies(accuracy_list):
	max_accuracy = 0.0
	best_clf = ''
	with open('accuracies.txt', 'w') as outfile:
		outfile.write("Classifier - Accuracy\n")
		for clf_name, accuracy in accuracy_list:
			line = "{} accuracy: {}\n".format(clf_name, accuracy)
			outfile.write(line)
			if (accuracy > max_accuracy):
				max_accuracy = accuracy
				best_clf = clf_name
		outfile.write("Best Classifier: {} accuracy {}\n".format(best_clf, max_accuracy))	


def write_submissions(predictions):
	#Writing to submission.csv
	with open('submissions.csv', 'w', newline='') as csvfile:
		fieldnames = ['PassengerId', 'Survived']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		passenger = 892
		for item in predictions: 
			writer.writerow({'PassengerId': passenger, 'Survived': item})
			passenger += 1
	print("File created successfully")


def train_classifier(clf, features_train, labels_train):
	clf.fit(features_train, labels_train)
	return clf


def make_predictions(clf, features_test):
	predictions = clf.predict(features_test)
	return predictions


def calc_accuracy(clf, predictions, labels_test):
	accuracy = accuracy_score(predictions, labels_test)
	return accuracy


# used to faclitate classifier testing
def test_classifier(clf, features_train, labels_train, features_test, labels_test):
	clf = train_classifier(clf, features_train, labels_train)
	predictions = make_predictions(clf, features_test)
	accuracy = calc_accuracy(clf, predictions, labels_test)
	print("accuracy:", accuracy)
	return accuracy


#####################################
#Main function call

if __name__ == '__main__':
	main()

#####################################