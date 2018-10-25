#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import ast
import math
import csv
import cv2
from matplotlib import pyplot as plt

dm = pd.read_csv('C:/Users/flono/Desktop/Cours_info/Informatique/L3/test_simplified.csv')
taillex=64
tailley=64


# In[2]:


dm['drawing'] = dm['drawing'].apply(ast.literal_eval)
#dm['drawing'].loc[0]


# In[3]:


def zipper(liste):
        return list(map(list, list(zip(*liste))))


# In[4]:


resConcat=dm[0:1]
def resres(dataFrameDraw):
    #global resConcat
    res=[]
    length=len(dataFrameDraw)
    for i in range(length):
        ss=dataFrameDraw.loc[i]
        #print(ss)
        dd = [zipper(liste) for liste in ss]
        print('ddd ::')
        print(dd)
        #str1 =  [v[0] for v in dd] 
        #print(i,"--------+-")
        #print(str1)
        res.append(dd)
    return res


# In[ ]:


dm['drawing'].loc[0]


# In[5]:


#def ParcTab():
# dkl=dm['drawing']
# for i in dkl:
#    zipper(liste) for liste in dkl.loc[i]
#    i++
#map(resres,dm['drawing'])

a=resres(dm['drawing'].loc[0:5])
a[1]

#type(resConcat['drawing'])


# In[6]:


img = np.zeros((512,512,3), np.uint8)
#cv2.line(img,(0,0),(511,511),(255,0,0),5)
#cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


# In[ ]:


#img = np.zeros((taillex,tailley,3), np.uint8)
#cv2.line(img,(0,0),(taillex,tailley),(0,255,0),5)
#plt.imshow(img)
#plt.title('my picture')
#plt.show()
#cv2.startWindowThread()
#cv2.namedWindow("zz")
#cv2.imshow("zz",img)
#cv2.waitKey()


# In[14]:


def createimage(cordes):
    def normaliseCoords(x,y):
        tmpx = int(round(x*taillex/255))
        tmpy = int(round(y*tailley/255))
        
        return tmpx, tmpy
    img = np.zeros((taillex, tailley,), np.uint8)
    for corde in cordes : 
        for i in range(len(corde)-1):
            print( tuple(reversed(normaliseCoords(corde[i][0], corde[i][1]))))
            cv2.line(img, normaliseCoords(corde[i][0], corde[i][1]), normaliseCoords(corde[i+1][0], corde[i+1][1]), (255), 1)
            
            
    return img


# In[17]:


img = createimage(a[3])
plt.imshow(img, cmap='gray')
plt.title('my picture')
plt.show()

