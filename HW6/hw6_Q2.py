#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import numpy as np


# In[2]:


get_ipython().run_line_magic('cd', 'D:\\StonyBrook\\Study\\Prob&Stats CSE544\\hw6')


# In[32]:


def process_data(dat_file):
    data = []
    with open(dat_file) as f:
        for line in f:
            data.append(line.split(', '))
    data = np.array(data,dtype=float)
    
    return data
    


# In[41]:


data_sigma3 = process_data('q2_sigma3.dat')
data_sigma100 =   process_data('q2_sigma100.dat')     


# In[47]:


def get_estimates(sigma, X , a, b_square):
    se_square = (sigma**2)/len(X)
    X_bar = np.mean(X)
    
    # calculating x and y
    denominator = b_square + se_square
    x = (b_square*X_bar + se_square*a)/denominator
    y_square = (b_square*se_square)/denominator
    
    return x,y_square
    


# In[48]:


table = []
sigma = 3
prior_mean = 0
prior_var = 1
for row_data in data_sigma3:
    prior_mean , prior_var = get_estimates(sigma,row_data,prior_mean,prior_var)
    table.append([prior_mean , prior_var])


# In[67]:


def plot_normals(table):
    for estimate in table:
        mu,variance = estimate[0], estimate[1]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label = 'mu: %.2f   , Var: %.2f ' %(mu,variance))
    plt.legend(loc="upper left")
    plt.title('Normal curves of estimates')
    plt.show()


# In[72]:


print(table)
plot_normals(table)


# In[73]:


table = []
sigma = 100
prior_mean = 0
prior_var = 1
for row_data in data_sigma100:
    prior_mean , prior_var = get_estimates(sigma,row_data,prior_mean,prior_var)
    table.append([prior_mean , prior_var])


# In[74]:


print(table)
plot_normals(table)

