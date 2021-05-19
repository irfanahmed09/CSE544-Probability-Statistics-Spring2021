#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('cd', 'D:\\StonyBrook\\Study\\Prob&Stats CSE544\\hw5')


# In[158]:


data = pd.read_csv('data_q4_1.csv')


# In[104]:


data


# In[159]:


data_stroke = data['avg_glucose_level'][data['stroke'] ==1]
data_nonstroke = data['avg_glucose_level'][data['stroke'] ==0]
data = data.drop(columns = ['stroke']).to_numpy()  


# In[160]:


print(np.mean(data_stroke) , np.mean(data_nonstroke))


# In[161]:


T_obs = abs(np.mean(data_stroke) - np.mean(data_nonstroke))
print("T_observed is : " + str(T_obs))


# In[164]:


def get_Ti(n_perm, data, n1):
    T = []
    for i in range(n_perm):
        permute = np.random.permutation(len(data))
        D1 = data[permute[:n1]]
        D2 = data[permute[n1:]]
        T.append(abs(np.mean(D1) - np.mean(D2)))
    
    return np.array(T)


# In[171]:


def get_p_value(T,T_obs):
    count = np.sum(T > T_obs)
    p_val = count/len(T)
    return p_val


# In[173]:


T_200 = get_Ti(200, data, len(data_stroke))
p_200 = get_p_value(T_200, T_obs)


# In[172]:


T_1000 = get_Ti(1000, data, len(data_stroke))
p_1000 = get_p_value(T_1000, T_obs)


# In[174]:


print("P value for n=200 : " + str(p_200))
print("P value for n=1000 : " + str(p_1000))


# In[175]:


p_val = 0.05 
print(" Both p values are less than 0.05. Hence reject H0 in both cases.")
print("\n Implies that people getting stroke tend to have the different glucose level as people who do not get stroke")


# In[177]:


data_2 = pd.read_csv('data_q4_2.csv')


# In[179]:


data_2_male = data_2['age'][data_2['gender'] =='Male'].to_numpy()
data_2_female = data_2['age'][data_2['gender'] =='Female'].to_numpy()
data_2 = data_2.drop(columns = ['gender']).to_numpy()


# In[180]:


T_obs = abs(np.mean(data_2_male) - np.mean(data_2_female))
print("T_observed is : " + str(T_obs))


# In[181]:


T_1000 = get_Ti(1000, data_2, len(data_2_male))
p_1000 = get_p_value(T_1000, T_obs)


# In[182]:


print("P value for n=1000 : " + str(p_1000))


# In[183]:


print("P-value is greater than 0.05. Implies accept H0. i.e.,  female  patients  get a  stroke at  the  same age as male  patients")


# In[185]:


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
    
    return X, Y


# In[199]:


def get_eCDF(X):
    n = len(X)
    Srt = sorted(X)
    delta = .1
    X = []
    Y = [0]
    for i in range(0,n):
        X = X + [Srt[i]]
        Y = Y + [Y[len(Y)-1]+(1/n)]
    Y = Y + [1]
    
    
    return X,Y


# In[211]:


X1,Y1 = plot_eCDF(data_2_male)
X2,Y2 = plot_eCDF(data_2_female)
X3,Y3 = [57.0,57.0] , [0.10185185185185185 , 0.2127659574468086]


# In[215]:


import matplotlib.pyplot as plt
plt.figure('eCDF')
plt.plot(X1, Y1 ,label='eCDF Male Age')
plt.plot(X2, Y2 ,label='eCDF Female Age')
plt.plot(X3,Y3, color= 'red', label=' Max difference = 0.1109')
#plt.scatter(sorted(X1), [0]*len(X1), color='red', marker='x', s=100, label='Age sample of Male')
#plt.scatter(sorted(X2), [0]*len(X2), color='green', marker='o', s=100, label='Age sample of Female')
plt.xlabel('x')
plt.ylabel('Pr[X<=x]')
plt.legend(loc="upper left")
plt.grid()
plt.show()


# In[202]:


X1,Y1 = get_eCDF(data_2_male)
X2,Y2 = get_eCDF(data_2_female)
 
Table = np.zeros((len(X1),6))


# In[209]:


tot_max = -1
for i in range(len(Table)):
    Table[i,0] = Y1[i]
    Table[i,1] = Y1[i+1]
    index1 = [idx for idx, val in enumerate(X2) if val >= X1[i]]
    index2 = [idx for idx, val in enumerate(X2) if val < X1[i]]
    Table[i,2] = Y2[index2[-1]]
    Table[i,3] = Y2[index1[0]]
    Table[i,4] = abs( Table[i,0] - Table[i,2])
    Table[i,5] = abs(Table[i,1] - Table[i,3])
    cmax = max(Table[i,4], Table[i,5])
    if cmax > tot_max:
        tot_max = cmax
        x1_max = X1[i]
        y1_max = Table[i,0]
        y2_max = Table[i,2]


# In[210]:


print(tot_max, x1_max, y1_max,  y2_max)

