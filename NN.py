##########################
# This code is an adaptation of Jose Portilla's code found here:
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# Check out his blog!
# by Python Noob on 10/6/2018
##########################

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

'''
HAR dataset available here:
http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

There was some cleaning done in R, found here:
https://rstudio-pubs-static.s3.amazonaws.com/291850_859937539fb14c37b0a311db344a6016.html
'''


os.chdir(        your path      + '/UCI HAR Dataset')
HAR = pd.read_csv('final_data.csv')
X = HAR.drop('lable', axis = 1)
y = HAR['lable']

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=300)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

confusion = confusion_matrix(y_test, predictions)
accuracy = np.trace(confusion)/len(X_test)

print('Accuracy: ' + str(float("{0:.2f}".format(100*accuracy))) + '%')
