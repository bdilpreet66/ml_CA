# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_excel("DATA-FINAL.xls")

# Data Preprocessing

# Understanding the data
dataset.describe()
dataset.isnull().sum()/dataset.shape[0]*100
len(dataset['Course'].unique())
len(dataset['MHRDName'].unique())
len(dataset['ScholarType'].unique())
len(dataset['Direction'].unique())
len(dataset['Gender'].unique())
len(dataset['Medium'].unique())
len(dataset['CourseType'].unique())
len(dataset['ProgramType'].unique())


# Data Extraction
y = dataset.iloc[:,[3]].values
X = dataset.iloc[:,[4,5,6,7,8,10,11,12,13,16,17,18,19,20,21]].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)




from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)



























