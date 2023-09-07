import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from subprocess import check_output
print(check_output(["ls", "../dataset"]).decode("utf8"))

train_df = pd.read_csv("../dataset/train.csv",header=0)
submit_test_df = pd.read_csv("../dataset/test.csv",header=0)

train_data = train_df.values
X_train, X_test, y_train, y_test = train_test_split(train_data[0:,1:], train_data[0:,0], test_size=0.2, random_state=0)
submit_test_data = submit_test_df.values

# Random Forest Classifier
forest = RandomForestClassifier(n_estimators = 500)
forest = forest.fit(X_train,y_train)
forest_output = forest.predict(X_test)
print("Random Forest with n_estimators:500")
print(accuracy_score(y_test, forest_output)) # 0.9663095238095238

forest = RandomForestClassifier(n_estimators = 5000)
forest = forest.fit(X_train,y_train)
forest_output = forest.predict(X_test)
print("Random Forest with n_estimators:5000")
print(accuracy_score(y_test, forest_output)) # 0.9660714285714286

# Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train,y_train)
gradient_output = clf.predict(X_test)  
print("GradientBoostingClassifier")
print(accuracy_score(y_test, gradient_output)) #0.7111904761904762

# MLP Classifier
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1)
clf.fit(X_train, y_train)   
neural_output = clf.predict(X_test)
print("sgd")
print(accuracy_score(y_test, neural_output)) # 0.925952380952381

clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,), random_state=1)
clf.fit(X_train, y_train)   
neural_output = clf.predict(X_test)
print("lbfgs")
print(accuracy_score(y_test, neural_output)) # 0.945952380952381

# Saving output
output = forest_output
predictions_file = open("forest_output.csv", "w")
open_file_object = csv.writer(predictions_file)
ids = range(forest_output.__len__())
ids = [x+1 for x in ids]
open_file_object.writerow(["ImageId", "Label"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Saved "forest_output" to file.')