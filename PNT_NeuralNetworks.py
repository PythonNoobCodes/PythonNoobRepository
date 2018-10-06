##########################
# This code is an adaptation of Jose Portilla's code found here:
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# Check out his blog!
# This iteration by Python Noob on 10/6/2018
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

'''
MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, 
alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, 
learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
random_state=None, tol=0.0001, verbose=False, warm_start=False, 
momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
n_iter_no_change=10)[source]
'''

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

confusion = confusion_matrix(y_test, predictions)
accuracy = np.trace(confusion)/len(X_test)

print('Accuracy: ' + str(float("{0:.2f}".format(100*accuracy))) + '%')
