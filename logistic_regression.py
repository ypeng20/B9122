# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 21:09:53 2014

@author: Yash Kanoria
"""

from sklearn import linear_model as lm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data=np.loadtxt("e_car_data.csv", delimiter=',', usecols=(7,8), skiprows=1)
print("data.shape =",data.shape, "\n data[:10, :]=\n", data[:10, :],"\n")

#APRs offered
x=data[:,1:2]
print("x[:10]=\n",x[:10])
print("x has length="+str(len(x)))
#We need x to be a 2D array, hence we do data[:,1:2] not data[:,1]

#Accepted or not
y=(data[:,0]>0)
print("y[:10]=",y[:10])
print("Fraction of loans accepted is", '%.3f'% np.mean(y))

#Create an object of class LogisticRegression
logistic=lm.LogisticRegression()

#Use the fit member function to fit a logistic regression
logistic.fit(x,y)

#Output the intercept and coefficient found here
print("intercept = ", '%.3f'% logistic.intercept_, "coefficient = ", '%.3f'% logistic.coef_)

ypred=logistic.predict_proba(x)

#A logistic classifier thresholds the prediction probability at 0.5 by default
print("Error probability of logistic classifier(in-sample):", '%.3f'%(1-logistic.score(x,y)))
print("RMSE of logistic prediction of probability is: ", '%.3f'%np.std(y-ypred[:,1]))

#Plot the ground truth vs the predicted probability of acceptance using pyplot
matplotlib.rcParams.update({'font.size': 25})
plt.figure(figsize=(10,5))
plt.scatter(ypred[:,1], y, marker = 'x', s = 120.)
plt.xlim(0, 1)
plt.ylim(-0.2, 1.2)
plt.yticks([0,1])
plt.xlabel('Predicted probability of acceptance')
plt.ylabel('Accepted (1) or not (0)')
plt.show()

