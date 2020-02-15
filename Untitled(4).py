#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

# Scientific and vector computation for python
import numpy as np
import pandas as pd
# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces
# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


data =pd.read_csv("house_prices_data_training_data.csv")
train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
X=np.array (train[['id','bedrooms','bathrooms','floors','view','condition']])
y = np.array(train['price'])
z=np.array(train[['bedrooms']])
m=y.size
print(y.shape)


# In[7]:


def plotData(z, y):
   
    fig = pyplot.figure()  # open a new figure
    
    pyplot.plot(z, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Price')
    pyplot.xlabel('bedrooms')
plotData(z, y)


# In[8]:


def  featureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = ( X - mu ) / sigma
    

    return X_norm, mu, sigma


# In[9]:


X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# In[10]:


def computeCostMulti(X, y, theta):
   
    m = y.shape[0]
    J = (1/(2*m))*np.dot(np.transpose(np.dot(X,theta)-y),np.dot(X,theta)-y)
    

    return J


# In[11]:


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = y.shape[0]
    theta = theta.copy()
    J_history = [] 
    for i in range(num_iters):
        alphabym=alpha/m
        sumofh0x=np.dot(X,theta)
        theta=theta-((alpha/m)*(np.dot(X.T,sumofh0x-y)))
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history


# In[12]:


alpha = 0.01
num_iters = 400
X = np.concatenate([np.ones((X.shape[0], 1)), X_norm], axis=1)
theta = np.zeros(7)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[ ]:




