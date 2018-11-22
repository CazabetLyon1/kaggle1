
#on commence par importe tt les bibliothéque dont on a besoin
import os
from glob import glob
import re
import ast
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw 
from tqdm import tqdm
from dask import bag

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#commencer par cree un fichier pr metrre tt dedans
dm = pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/RC4/circle.csv')
drr= dm.loc[0:2]

#une fonction qui parcours notre dossier RC4 et renvoie un fichier csv avec 5000 premiere ligne de chaque fichier
#draw
def ParcourDirectory():
    global drr
    i=0
    for element in os.listdir('C:/Users/chaoui/Desktop/LifProjet/RC4/'):
        dd= pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/RC4/'+element)
        drr=drr.append(dd.loc[0:5000])

ParcourDirectory()

#on transforme le resultats quon a eu en un fichier csv
drr.to_csv('C:/Users/chaoui/Desktop/LifProjet/test.csv')

#on relie le fichier dans une variable
dmm=pd.read_csv('C:/Users/chaoui/Desktop/LifProjet/test.csv')
dmm

#on applique la fct ast eval pour quon puisse traiter les drawing en tant que liste et non une chaine de caractére
dmm['drawing'] = dmm['drawing'].apply(ast.literal_eval)

#fonction pr transformer la liste de vecteur en en couple (x,y)
def zipper(liste):
        return list(map(list, list(zip(*liste))))

#une fonction resres qui apelle cette fonction sur tt un tableau et renvoie en resultats le tableau modifier
def resres(dataFrameDraw):
    res=[]
    length=len(dataFrameDraw)
    for i in range(length):
        ss=dataFrameDraw.iloc[i]
        dd = [zipper(liste) for liste in ss]
        res.append(dd)
    return res

#une fonction qui apelle zipper mais uniquement pour une seul ligne du tableau
def resres1(dataFrame):
    ss=dataFrame
    dd = [zipper(liste) for liste in ss]
    return dd

#une fonction pour cree notre image , prend en parametre la ligne du tableau le parcour puis dessigne sur une image blache
#des trait de entre chaque couple et renvoie le resultats en tant que tableau contenant tt les pixel de l'image grace a la 
#fonction predefinie open_cv qui fait le job 
def createimage(cordes):
    def normaliseCoords(x,y):
        tmpx = int(round(x*taillex/255))
        tmpy = int(round(y*tailley/255))
        return tmpx, tmpy

    img = np.zeros((taillex, tailley,), np.uint8)
    for corde in cordes : 
        for i in range(len(corde)-1):
            cv2.line(img, normaliseCoords(corde[i][0], corde[i][1]), normaliseCoords(corde[i+1][0], corde[i+1][1]), (255), 1)
    
    return img

#ici on va dans le train_simplified et on commence par remplacer tt les vide dans les noms par des tirets 
classfiles = os.listdir('../input/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #ajoute tt les underscore

#declarer quel que constante, l'image je l'utilisais a 32 par 32 comme ca j'ai un meilleur resulats tt en ayant un temps moins important de #calcul "d'apres les test que j'ai fais"
num_classes = 340    #340 valeur max le nombre de fichier dans RC4 
imheight, imwidth = 32, 32  
ims_per_class = 2000  #max?

