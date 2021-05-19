#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


# In[4]:


def plot_eCDF(X):
    n = len(X)
    Srt = sorted(X)
    delta = .1
    X = [min(Srt)-delta]
    Y = [0]
    for i in range(0, n):
        X = X + [Srt[i], Srt[i]]
        Y = Y + [Y[len(Y)-1], Y[len(Y)-1]+(1/n)]
    X = X + [max(Srt)+delta]
    Y = Y + [1]
    
    plt.figure('eCDF')
    plt.plot(X, Y ,label='eCDF')
    plt.scatter(Srt, [0]*n, color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    return X,Y


# In[92]:


X1 = np.random.randint(1,100,10)
X2 = np.random.randint(1,100,100)
X3 = np.random.randint(1,100,1000)
print("Q3.b)")
A,B = plot_eCDF(X1)
print(A,B)
print("Q3.b)")
A,B = plot_eCDF(X2)
print("Q3.b)")
A,B = plot_eCDF(X3)


# Q3.b) Observation:- With increasing samples(i.e, 10 to 1000) we see that the estimated CDF behaves as true CDF. This implies that when n tends to infinity, eCDF tends to true CDF. 

# In[89]:


def modified_eCDF(XY):
    Y = []
    X = []
    XY.sort(axis = 1)
    delta = .1
    m,n = XY.shape
    Y = [0]
    X = [np.min(XY)-delta]
    for i in range(m):
        for j in range(n):
            Y = Y + [ Y[len(Y)-1] , 1/(m*n) + Y[len(Y)-1] ]
            X = X + [XY[i,j], XY[i,j]]
    
    X, Y = np.array(X), np.array(Y)
    Y = Y
    plot_XY(X,Y,m)
    return X,Y


# In[42]:


def plot_XY(X,Y,m):
    n = len(X)
    X.sort()
    #Y.sort()
    plt.figure('Avg F_hat')
    plt.plot(X, Y , label='average F_hat for x')
    plt.scatter(X, [0]*n, color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Avg F_hat')
    plt.title('Avg F_hat with %d x %d samples. Sample mean = %.2f.' % (m,10, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


# In[93]:


XY1 = np.random.randint(1,100,(10,10))
XY2 = np.random.randint(1,100,(100,10))
XY3 = np.random.randint(1,100,(1000,10))

print("Q3.d)")
modified_eCDF(XY1)
print("Q3.d)")
modified_eCDF(XY2)
print("Q3.d)")
modified_eCDF(XY3)


# Q3.d) With increasing samples of X, we see that the average eCDF behaves as true CDF.

# In[9]:


get_ipython().run_line_magic('cd', 'C:\\Users\\irfan\\Downloads')
data = pd.read_csv('a3_q3.csv', header=None).to_numpy()



# In[102]:


def CI_plot(X):
    n = len(X)
    Srt = sorted(X)
    delta = .1
    X = [min(Srt)-delta]
    Y = [0]
    Z1 = [0]
    Z2 = [0]
    for i in range(0, n):
        X = X + [Srt[i], Srt[i]]
        f_hat = Y[len(Y)-1]
        Y = Y + [f_hat, f_hat +(1/n)]
        Z1 = Z1 + [f_hat + np.sqrt(f_hat* (1-f_hat)/n), f_hat + np.sqrt(f_hat* (1-f_hat)/n) ]
        Z2 = Z2 + [f_hat - np.sqrt(f_hat* (1-f_hat)/n), f_hat - np.sqrt(f_hat* (1-f_hat)/n) ]
    X = X + [max(Srt)+delta]
    Y = Y + [1]
    
    
    Z1 = Z1 + [1]
    Z2 = Z2 + [1]
    
    DKW_bound = np.sqrt(np.log(2/0.05) * (1/(2*n)))
    #print(DKW_bound)
    
    Y = np.array(Y)
    
    plt.figure('eCDF')
    plt.plot(X, Y ,label='eCDF')
    plt.scatter(Srt, [0]*n, color='red', marker='x', s=100, label='samples')
    
    plt.plot(X, Z1 ,label = '95% Normal CI bounds', color = 'gray')
    plt.plot(X,Z2 , color = 'gray')
    
    plt.plot(X, Y + DKW_bound ,label = '95% DKW CI bounds', color = 'orange')
    plt.plot(X, Y - DKW_bound , color = 'orange')
    
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    
    plt.show()
    


# In[99]:


CI_plot(data)


# In[103]:


CI_plot(data)

