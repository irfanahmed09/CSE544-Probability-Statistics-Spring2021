#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from scipy.stats import norm
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt


# In[33]:


get_ipython().run_line_magic('cd', 'C:\\Users\\irfan\\Downloads')


# In[35]:


def normal_kde(x,h,D):
    n = len(D)
    u = (x - D)/h
    K_u = norm.pdf(u)
    f_hat_x = np.sum(K_u)/(n * h)
    
    return f_hat_x

def uniform_kde(x,h,D):
    n = len(D)
    u = (x - D)/h
    K_u = np.zeros(n)
    for i in range(len(u)):
        if(u[i] >= -1 and u[i] <= 1):
            K_u[i] = 0.5
        else:
            K_u[i] = 0
    
    f_hat_x = np.sum(K_u)/(n * h)
    
    return f_hat_x

def triangular_kde(x,h,D):
    n = len(D)
    u = (x - D)/h
    K_u = np.zeros(n)
    for i in range(len(u)):
        if(abs(u[i]) <= 1):
            K_u[i] = 1 - abs(u[i])
        else:
            K_u[i] = 0
    
    f_hat_x = np.sum(K_u)/(n * h)
    
    return f_hat_x


# In[36]:


data = pd.read_csv('a3_q7.csv').to_numpy()
X = np.arange(0,1.01,0.01)
D = data[:,1]
sample_mean = np.mean(X)
sample_var = np.var(X)


# In[49]:


def estimate(X,h,D,function_kde):
    Y = []
    for i in range(len(X)):
        Y.append(function_kde(X[i],h,D))
    Y = np.array(Y)
    
    return Y


# In[39]:


def deviation(true,pred):
    return 100 * abs(pred - true)/ true


# In[97]:


h = [0.0001,  0.0005,  0.001,  0.005,  0.05]

for i in range(len(h)):
    Y = estimate(X,h[i],D,normal_kde)
    #print("Deviation from mean : %.4f for h: %.4f" %(deviation(0.5,a), h[i]))
    print("For h: %.4f " %(h[i]))
    print("Estimated Mean: %.2f , Estimated Variance: %.2f" % (np.mean(Y), np.var(Y)))
    print("Percentage Deviation from Mean : %.4f " %(deviation(0.5,np.mean(Y))))
    plt.plot(X,Y,color = 'blue',label = '(mu : %.2f, std: %.2f) h: %.4f' % (np.mean(Y),np.sqrt(np.var(Y)), h[i]) )
    
    plt.xlabel('x')
    plt.ylabel('PDF Pr[X=x]')
    
    
    true_mu, true_sigma = 0.5, 0.1
    
    #x = np.linspace(true_mu - 3*true_sigma, true_mu + 3*true_sigma, 100)  
    plt.plot(X, stats.norm.pdf(X, true_mu, true_sigma),color = 'orange', label = 'Normal((%.2f,%.2f)'  %(true_mu,true_sigma))
    
    
    plt.show()


#plt.legend(loc="upper right",bbox_to_anchor=(1.5, 1))




print("Best Value of h: 0.05")


# In[98]:


h = [0.0001,  0.0005,  0.001,  0.005,  0.05]

for i in range(len(h)):
    Y = estimate(X,h[i],D,uniform_kde)
    #print("Deviation from mean : %.4f for h: %.4f" %(deviation(0.5,a), h[i]))
    print("For h: %.4f " %(h[i]))
    print("Estimated Mean: %.2f , Estimated Variance: %.2f" % (np.mean(Y), np.var(Y)))
    print("Percentage Deviation from Mean : %.4f " %(deviation(0.5,np.mean(Y))))
    plt.plot(X,Y,color = 'blue',label = '(mu : %.2f, std: %.2f) h: %.4f' % (np.mean(Y),np.sqrt(np.var(Y)), h[i]) )
    
    plt.xlabel('x')
    plt.ylabel('PDF Pr[X=x]')
    plt.plot(X,Y,label = 'Normal(%.2f,%.2f) h: %.4f' % (np.mean(Y),np.sqrt(np.var(Y)), h[i]) )
    
    true_mu, true_sigma = 0.5, 0.1
    
    #x = np.linspace(true_mu - 3*true_sigma, true_mu + 3*true_sigma, 100)  
    plt.plot(X, stats.norm.pdf(X, true_mu, true_sigma), label = 'Normal((%.2f,%.2f)'  %(true_mu,true_sigma))
    
    plt.show()




#plt.legend(loc="upper right",bbox_to_anchor=(1.5, 1))


print("Best Value of h: 0.05")


# In[96]:


h = [0.0001,  0.0005,  0.001,  0.005,  0.05]

for i in range(len(h)):
    Y = estimate(X,h[i],D,triangular_kde)
    #print("Deviation from mean : %.4f for h: %.4f" %(deviation(0.5,a), h[i]))
    print("For h: %.4f " %(h[i]))
    print("Estimated Mean: %.2f , Estimated Variance: %.2f" % (np.mean(Y), np.var(Y)))
    print("Percentage Deviation from Mean : %.4f" %(deviation(0.5,np.mean(Y))))
    plt.plot(X,Y,color= 'blue',label = '(mu : %.2f, std: %.2f) h: %.4f' % (np.mean(Y),np.sqrt(np.var(Y)), h[i]) )
    
    plt.xlabel('x')
    plt.ylabel('PDF Pr[X=x]')
    plt.plot(X,Y,label = 'Normal(%.2f,%.2f) h: %.4f' % (np.mean(Y),np.sqrt(np.var(Y)), h[i]) )
    true_mu, true_sigma = 0.5, 0.1
    
    #x = np.linspace(true_mu - 3*true_sigma, true_mu + 3*true_sigma, 100)  
    plt.plot(X, stats.norm.pdf(X, true_mu, true_sigma), label = 'Normal((%.2f,%.2f)'  %(true_mu,true_sigma))
    
    plt.show()


print("Best Value of h: 0.05")

