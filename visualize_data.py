#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from pykrige.ok import OrdinaryKriging


# In[5]:


import math


# In[6]:


def spi(H,Perm,S,Por):
    Spi = (H/20)*math.pow((Perm/1),0.8)*math.pow((S/0.5),2)*math.pow((Por/0.3),1.5)
    return Spi


# In[131]:


np.sort(np.random.random_integers(25,35 ,100) * 0.01)


# In[192]:


Porosity = np.sort(np.random.random_integers(25,35 ,100))
p = Porosity*0.01
    


saturation = np.sort(np.random.random_integers(60,80,100))

s= saturation*0.01

height = np.sort(np.random.random_integers(15,35,100))

permiability = np.sort(np.random.random_integers(600,3000,100))

per = permiability *0.001

a= []
for i in range(100):
    k=spi(height[i],per[i],s[i],p[i])
    a.append(k)
    
sspi=np.array(a)

Max = np.max(sspi)
Min = np.min(sspi)
def normalize(a):
    b=[]
    for i in range(len(a)):
        normal = a[i]/5
        b.append(normal)
    return np.array(b)

def Normalize(a):
    b= []
    Max = np.max(a)
    Min = np.min(a)
    for i in range(len(a)):
        normal = (a[i]-Min)/(Max-Min)
        b.append(normal)
    return np.array(b)
normalizespi = normalize(sspi)
print(normalizespi)
print("깊이,투과도, 공극률, 포화도")
print(height,per,p,s)

y=normalizespi


# In[135]:


normalizespi = normalize(sspi)
print("spi")
print(sspi)
print("normalizespi")
print(normalizespi)
print("깊이,투과도, 공극률, 포화도")
print(height,per,p,s)
q= Normalize(sspi)
print(q)
y=normalizespi


# In[201]:


def conv(a):
    one =a[:25]
    two =a[25:50]
    three=a[50:75]
    four=a[75:]
    np.random.shuffle(one)
    np.random.shuffle(two)
    np.random.shuffle(three)
    np.random.shuffle(four)
    one_n = one.reshape(5,5)
    two_n = two.reshape(5,5)
    three_n = three.reshape(5,5)
    four_n = four.reshape(5,5)
    two_four = np.vstack([four_n,two_n])
    three_one = np.vstack([three_n,one_n])
    complete = np.hstack([three_one,two_four])
    return complete


    
    


# In[220]:


new_sat = conv(s)
new_depth = conv(height)
new_perm = conv(per)
new_porosity= conv(p)
print(new_porosity)


# In[208]:



def visualize_data_with_O_k(input_data):
    output_data=[]
    for i in range(10):
        for j in range(10):
        
            k=[]
            k.append(lat[i])
            k.append(lon[j])
            k.append(input_data[i][j])
            output_data.append(k)
    
    data = np.array(data)
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='gaussian',verbose=False, enable_plotting=False)
    z, ss = OK.execute('grid', gridx, gridy)
    print(z)
    plt.contourf(gridx, gridy, z)
    plt.colorbar()
    plt.xlabel('Normalized Latitude')
    plt.ylabel('Normalized Longitude')
    plt.show()

visualize_data_with_O_k(new_sat) 


# In[209]:


visualize_data_with_O_k(new_depth)


# In[212]:


visualize_data_with_O_k(new_perm)


# In[213]:


visualize_data_with_O_k(new_porosity)


# In[186]:


zq = pd.DataFrame(z)
df=zq.unstack().reset_index()
df.columns=["X","Y","Z"]
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()


# In[ ]:




