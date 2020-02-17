#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

# Scientific and vector computation for python
import numpy as np
import pandas as pd
# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces
# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def  featureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = ( X - mu ) / sigma
    

    return X_norm, mu, sigma


# In[4]:


def computeCostMulti(X, y, theta, lambda_):
   
    m = y.shape[0]
    J = (1/(2*m))*np.dot(np.transpose(np.dot(X,theta)-y),np.dot(X,theta)-y)+ ((lambda_/(2 * m))* np.sum(np.dot(theta, theta)))
    

    return J


# In[5]:


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = y.shape[0]
    theta = theta.copy()
    J_history = [] 
    for i in range(num_iters):
        alphabym=alpha/m
        sumofh0x=np.dot(X,theta)
        theta=theta*(1 - (alpha*lambda_)/m) -((alpha/m)*(np.dot(X.T,sumofh0x-y)))
        J_history.append(computeCostMulti(X, y, theta, lambda_))
    return theta, J_history


# In[13]:


alpha = 0.01
alpha2=0.003
num_iters = 150

theta = np.zeros(7)
data =pd.read_csv("house_data_complete.csv")
for i in range(3):

    train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
    X=np.array (train[['id','bedrooms','bathrooms','floors','view','condition']])
    y = np.array(train['price'])
    z=np.array(train[['bedrooms']])
    m=y.size
    X_norm, mu, sigma = featureNormalize(X)
    X = np.concatenate([np.ones((X.shape[0], 1)), X_norm], axis=1)
    lambda_ = 0.5
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
    theta2, J_history2 = gradientDescentMulti(np.power(X,2), y, theta,alpha2 , num_iters)
    theta3, J_history3 = gradientDescentMulti(np.power(X,3), y, theta, alpha2, num_iters)
    J= computeCostMulti(X, y, theta, lambda_)
    J2= computeCostMulti(np.power(X,2), y, theta, lambda_)
    J3= computeCostMulti(np.power(X,3), y, theta, lambda_)
    print("j1", J)
    print("j2" ,J2)
    print("j3" ,J3)
    
    pyplot.figure()
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2, label='h1')
    pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2, label='h2')
    pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2, label='h3')

    pyplot.legend()
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')


# In[7]:


def plotData(z, y):
   
    fig = pyplot.figure()  # open a new figure
    
    pyplot.plot(z, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Price')
    pyplot.xlabel('bedrooms')
plotData(z, y)


# In[ ]:




