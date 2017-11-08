'''
Feature setup for TitanicML
'''

import pandas as pd


def setup_data(data): 
	data["Title"] = 0

	for i in data:
		data['Title'] = data.Name.str.extract('([\w]+)\.', expand=True)
	
	# replaces unnecessary titles
	data['Title'] = data["Title"].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
			['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Mr','Mr','Mr','Mr','Mr'])
	data['Title'] = data["Title"].apply(lambda x: 'Mrs' if x == 'Dona' else x)

	# used to observe the average age for each Title
	#print(data.groupby('Title')['Age'].mean())
	data.loc[(data.Age.isnull()) & (data.Title == 'Master'), 'Age'] = 5
	data.loc[(data.Age.isnull()) & (data.Title == 'Miss'), 'Age'] = 22
	data.loc[(data.Age.isnull()) & (data.Title == 'Mr'), 'Age'] = 33
	data.loc[(data.Age.isnull()) & (data.Title == 'Mrs'), 'Age'] = 36
	data.loc[(data.Age.isnull()) & (data.Title == 'Other'), 'Age'] = 38
	data['Embarked'].fillna('S', inplace=True)

	data['Age_Range'] = 0
	data.loc[(data['Age'] <= 16, 'Age_Range')] = 0
	data.loc[(16 < data['Age']) & (data['Age'] <= 32), 'Age_Range'] = 1
	data.loc[(32 < data['Age']) & (data['Age'] <= 48), 'Age_Range'] = 2
	data.loc[(48 < data['Age']) & (data['Age'] <= 64), 'Age_Range'] = 3
	data.loc[(64 < data['Age']), 'Age_Range'] = 4

	data['Fare_Split'] = pd.qcut(data['Fare'], 4)
	#print(data.groupby(['Fare_Split'])['Survived'].mean())		
	# creates Fare_Range
	data['Fare_Range'] = 0
	data.loc[data['Fare'] <= 7.91, 'Fare_Range'] = 0
	data.loc[(7.91 < data['Fare']) & (data['Fare'] <= 14.45), 'Fare_Range'] = 1
	data.loc[(14.45 < data['Fare']) & (data['Fare'] <= 31), 'Fare_Range'] = 2
	data.loc[31 < data['Fare'], 'Fare_Range'] = 3

	data['Sex'] = data['Sex'].replace(['male','female'], [0,1])
	data['Embarked'] = data['Embarked'].replace(['S','C','Q'], [0,1,2])
	data['Title'] = data['Title'].replace(['Mr','Mrs','Miss','Master','Other'], [0,1,2,3,4])

	#print('All features setup successfully.')

	return data.drop(['Name','Age','Ticket','Fare','Cabin', 'Fare_Split','PassengerId'], axis=1)
