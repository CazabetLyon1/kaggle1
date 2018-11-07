
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
import ast
import math
import csv
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')

taillex=64
tailley=64

labeled_images = pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/test.csv')
images = dmm['drawing']
labels = dmm['word']
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# In[489]:


dm = pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/RC4/circle.csv')
drr= dm.loc[0:2]


# In[9]:


def ParcourDirectory():
 #with open('C:/Users/YB.DELL/Desktop/LifProjet/test.csv','a') as fd:
    global drr
    for element in os.listdir('C:/Users/YB.DELL/Desktop/LifProjet/RC4/'):
        dd= pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/RC4/'+element)
    print(element)
    #dg= pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/RC4/angel.csv')
    drr=drr.append(dd.loc[0:2])
    #fd.write(dd.loc[0:2])


# In[ ]:


ParcourDirectory()


# In[5]:


#dd= pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/RC4/airplane.csv')
#dg= pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/RC4/angel.csv')
#dd.append(dg.loc[0:2])
drr.to_csv('C:/Users/YB.DELL/Desktop/LifProjet/test.csv')


# In[6]:


dmm=pd.read_csv('C:/Users/YB.DELL/Desktop/LifProjet/test.csv')
dmm


# In[7]:


dmm['drawing'] = dmm['drawing'].apply(ast.literal_eval)


# In[9]:


def zipper(liste):
        return list(map(list, list(zip(*liste))))


# In[10]:


def resres(dataFrameDraw):
    res=[]
    length=len(dataFrameDraw)
    for i in range(length):
        ss=dataFrameDraw.iloc[i]
        dd = [zipper(liste) for liste in ss]
        res.append(dd)
    return res


# In[11]:


def resres1(dataFrame):
    ss=dataFrame
    dd = [zipper(liste) for liste in ss]
    return dd


# In[12]:


dmdm=resres1(dmm['drawing'].iloc[0])
dmdm


# In[13]:


a=resres(dmm['drawing'].iloc[0:5])
a[3]


# In[14]:


def createimage(cordes,cd):
    cc=[]
    def normaliseCoords(x,y):
        tmpx = int(round(x*taillex/255))
        tmpy = int(round(y*tailley/255))
        
        return tmpx, tmpy
    img = np.zeros((taillex, tailley,), np.uint8)
    for corde in cordes : 
        for i in range(len(corde)-1):
            #print( tuple(reversed(normaliseCoords(corde[i][0], corde[i][1]))))
            cc=cv2.line(img, normaliseCoords(corde[i][0], corde[i][1]), normaliseCoords(corde[i+1][0], corde[i+1][1]), (255), 1)
            #print(cc)
        cd.append(cc)    
    return img


# In[21]:


listeV=[]
img = createimage(a[3],listeV)
plt.imshow(img, cmap='binary')
plt.title('my picture')
plt.show()


# In[22]:


listeV[0][19]#==listeV[1][0]


# In[23]:


def draw_img(to_draw,a,b,listepoints):
    i=0
    cd=[]
    tt=[]
    a=resres(to_draw['drawing'].iloc[a:b])
    for i in range(len(a)):
        cd.append(tt)
    for i in range(len(a)):
        img = createimage(a[i],cd[i])
        listepoints.append(cd[i])
        plt.imshow(img, cmap='binary')
        plt.title(to_draw['word'][i])
        plt.show()  


# In[ ]:


ls=[]
draw_img(dmm,0,3,ls)


# In[89]:


n = 255
m = 255
a = [0] * n
for i in range(n):
    a[i] = [0] * m


# In[43]:


def créeListe(taillezboub):
    return [[0 for j in range(taillezboub)] for i in range(taillezboub)]


# In[415]:


img = np.zeros((10,10,3), np.uint8)
abab=cv2.line(img,(0,0),(511,511),(255),1)
plt.imshow(img, cmap='binary')
plt.title('my picture')
plt.show()
#abab


# In[266]:


list(abab)
listt=[]
for ligne in abab:
    #print(ligne)
    listt.append([point[0] for point in ligne])
print(listt)


# In[156]:


def replacell(Liste):
 for ligne in Liste:
    for element in range(len(ligne)):
        if ligne[element]>0:
            ligne[element]=1  
 return Liste            


# In[157]:


kdkd=[]
kdkd=replacell(listt)
kdkd


# In[249]:


listeV[0][2]


# In[366]:


listeV
listeRes=[]
for ligne in listeV:
    #print(ligne)
    for element in ligne:
        #print(element)
        listeRes.append(element)
    #listeRes.append([point[0] for point in ligne])
listeRes
#list(listeRes)
#type(listeRes[0])


# In[382]:


listeM=[]
for ligne in listeRes:
    listeM.append(ligne)
#    for point in ligne:
#            print(" ")
    #listeM.append([point[0] for point in ligne])
type(listeM[0])


# In[432]:


for l in listeV:
    listeM += l
listll=listeM.flatten
lll=np.concatenate(listeV, axis=0)
lll.flatten()
lll[0]


# In[428]:


ac=np.array(listeV).flatten()


# In[436]:


len(ac)


# In[10]:


def Pixel_Flatten(dataFrame):
        a=resres1(dataFrame)
        listeF=[]
        createimage(a,listeF)
        lsls=np.array(listeF).flatten()
        return lsls


# In[11]:


def Parcour_Tab_pixel(dataFrame):
    for i in range(len(dataFrame)):
        dataFrame[i]=Pixel_Flatten(dataFrame[i])


# In[18]:


temp=dmm['drawing'].iloc[0:]
#Parcour_Tab_pixel(images)
dmm


# In[1]:


dmm


# In[23]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values)
clf.score(test_images,test_labels)

