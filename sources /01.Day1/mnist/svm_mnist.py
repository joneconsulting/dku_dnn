import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
from sklearn.metrics import accuracy_score

# The digits dataset
train=pd.read_csv('../dataset/mnist_train_100.csv', header=None)
test=pd.read_csv('../dataset/mnist_test_10.csv', header=None)
# train=pd.read_csv('../dataset/mnist_train.csv', header=None)
# test=pd.read_csv('../dataset/mnist_test.csv', header=None)

data_train=train.iloc[:,1:].values
label_train=train.iloc[:,0].values

# data_test=data_test.values
data_test=test.iloc[:,1:].values
label_test=test.iloc[:,0].values

'''
data scaling	,  직접 scaling 하거나, StandardScaler, MinMaxScaler 이용
'''
# data_train[data_train>0]=1
# data_test[data_test>0]=1
scaler = preprocessing.MinMaxScaler()
data_train_scaled = scaler.fit_transform(data_train.astype(np.float32))
data_test_scaled = scaler.fit_transform(data_test.astype(np.float32))
data_train = data_train_scaled
data_test = data_test_scaled

# print(data_train)
# print(label_train)
'''
Create a classifier: a support vector classifier
'''
classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
#classifier = svm.SVC()

# We learn the digits on the first half of the digits
classifier.fit(data_train, label_train)

# Now predict the value of the digit on the second half:
#expected = label_test
predicted = classifier.predict(data_test)

'''print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))'''

#print(expected)
df=pd.DataFrame(predicted)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv',header=True)
print(predicted)

''' accuracy_score '''
print(accuracy_score(predicted, label_test))