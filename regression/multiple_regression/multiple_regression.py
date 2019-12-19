#!/opt/local/bin/python2.7

import math
from sklearn import linear_model, preprocessing
import numpy as np


### Data ###
# yrs education, avg hrs/week worked, weight (lbs)
xtrain_raw = np.asmatrix([
     [14, 40, 150]
    ,[16, 45, 220]
    ,[18, 55, 180]
    ,[18, 80, 190]
    ,[10, 20, 300]
    ])
# income
ytrain_raw = np.asarray([33000, 55000, 100000, 500000, 12000])  # income

xtest_raw = np.asmatrix([
     [13, 40, 175]
    ])
ytest_raw = np.asarray([40000])

# Normalize.
# xtrain = np.transpose(preprocessing.normalize(np.transpose(xtrain_raw), norm='max'))
# ytrain = np.transpose(preprocessing.normalize(np.transpose(ytrain_raw), norm='max'))
# xtest = np.transpose(preprocessing.normalize(np.transpose(xtest_raw), norm='max'))

# xtrain = preprocessing.normalize(xtrain_raw, norm='max', axis=0)
# ytrain = np.transpose(preprocessing.normalize(ytrain_raw, norm='max'))
# xtest = preprocessing.normalize(xtest_raw, norm='max', axis=0)

xtrain = xtrain_raw
ytrain = ytrain_raw
xtest  = xtest_raw
ytest  = ytest_raw

# Data debug.
print( "xtrain: " + str(xtrain) )
print( "ytrain: " + str(ytrain) )
print( "xtest: " + str(xtest) )


### Modeling ###
# Create linear model.
lm = linear_model.LinearRegression()

# Train.
lm.fit(xtrain, ytrain)
lms = lm.score(xtrain, ytrain)


### Results ###
# Print the coefs and intercept.
print( 'Intercept: ' + str(lm.intercept_) )
print( 'Coefs: ' + str(lm.coef_) )
# Print the prediction.
predicted = lm.predict(xtest)
print( "Prediction: " + str(predicted) )

# Print SSE & RMSE (root mean sum-of-squared errors).
sse = np.mean( (predicted - ytest) ** 2)
print( "SSE: " + str(sse) )
print( "RMSE: " + str(math.sqrt(sse)) )
