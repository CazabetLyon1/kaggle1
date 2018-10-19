#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import numpy as np
import os
import ast
import math
import matplotlib as plt
import csv

dm = pd.read_csv('C:/Users/flono/Desktop/Cours_info/Informatique/L3/test_simplified.csv')
#dm


# In[ ]:


dm['drawing'] = dm['drawing'].apply(ast.literal_eval)


# In[6]:


dm['drawing'].loc[0]


# In[8]:


def zipper(liste):
        return list(map(list, list(zip(*liste))))


# In[102]:


resConcat=dm[0:1]

def resres(terter):
    global resConcat
    length=len(terter)
    for i in range(length):
        ss=terter.loc[i]
        dd = [zipper(liste) for liste in ss]
        str1 = pd.Series( (v[0] for v in dd) )
        resConcat['drawing']=resConcat['drawing'].append(str1)
        i=i+1
    print(i)
    #print(resConcat)
    print(type(str1))
    #return resConcat
#resConcat


# In[103]:


#def ParcTab():
# dkl=dm['drawing']
# for i in dkl:
#    zipper(liste) for liste in dkl.loc[i]
#    i++
#map(resres,dm['drawing'])
resres(dm['drawing'])
#type(resConcat['drawing'])


# In[57]:


resConcat

