
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

#juste un ptit affichage d'image
listeV=[]
img = createimage(a[6])
plt.imshow(img, cmap='binary')
plt.title('my picture')
plt.show()

#une fonction qui fait tt ltaf qui transforme,convertion,dessine 
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

#fonction qui prend une ligne la transforme, puis la convertie , puis recupeere les pixel apres avoir dessiner l'image
def Pixel_Flatten(dataFrame):
        a=resres1(dataFrame)
        listeF=[]
        listeF=createimage(a)
        lsls=np.array(listeF).flatten()
        return lsls

#la fonction qui prend en charge un tableau et fait tt le taf a faire 
def Parcour_Tab_pixel(dataFrame):
    for i in range(len(dataFrame)):
        dataFrame[i]=Pixel_Flatten(dataFrame[i])

#on appele cette fct sur notre tableau dmm qui contient tt les lignes 
Parcour_Tab_pixel(dmm['drawing'])
dmm['drawing']

#dans un premier temps on utilise pas un reseaux neuronal, on utilise plustot au depart un model
#predefinit avec sklearn (scikitlearn)

#on commence a instancier les model 
images = dmm['drawing']
labels = dmm['word']
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#on commence mettre nos list au bon format 
lss=list(train_images)
res=[list(x) for x in lss ]
lss1=list(test_images)
res1=[list(x) for x in lss1 ]

#la on passe les resultats au bn format
ress=list(train_labels)
ress1=list(test_labels)

#on apelle la fonction qui trouve le resultats 
clf = svm.SVC()
clf.fit(res,ress)
clf.score(res1,ress1)

#ca va nous renvoyer la valeur 0.096 ce qui est apein 10% et donc ne represente un resultats tres tres faible
#on a ce meme resultats si on devine les noms des draw au pif
#et donc on essaye une autre methode avec notre propre cnn



///////////////////////////////////////////cnn propre a nous//////////////////////////////////////////////////



#ici on va dans le train_simplified et on commence par remplacer tt les vide dans les noms par des tirets 
classfiles = os.listdir('../input/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #ajoute tt les underscore

#declarer quel que constante, l'image je l'utilisais a 32 par 32 comme ca j'ai un meilleur resulats tt en ayant un temps moins important de #calcul "d'apres les test que j'ai fais"
num_classes = 340    #340 valeur max le nombre de fichier dans RC4 
imheight, imwidth = 32, 32  
ims_per_class = 2000  #max?

#ici apres avoir fait mon test avec ma premier fonction pour dessiner les images , apres avoir crash mon pc... 
#j'ai chercher sur internet et apparament d'apres les erreur que j'avais ca m'afficher 
#ma fonction n'etait pas tres memory-friendly et donc j'ai du redefinire une fonction pour dessiner l'image 
#qui est un peu plus rapide mais aussi ne devorer pas la memoire(stack overflow ma aider) 

def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    #on redimensionne l'image au parametre de taille qu'on a         
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.


#ici c'est une fonction pr definir les train_images et les train_labels quand je l'utilise j'ai un meilleur 
#resultats , c'est une methode de monsieur fracois chollet que j'ai reprise a patire du tuto qu'il a mis en ligne 
#pour le digit reconizer , c'est la personne qui a cree la bibliotheque keras 
#en gros ici il remplace ce que train_data_split fait 
train_grand = []
#recupere tt les fichier avec .csv a la fin
class_paths = glob('../input/train_simplified/*.csv')
for i,c in enumerate(tqdm(class_paths[0: num_classes])):
    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)
    train = train[train.recognized == True].head(ims_per_class)
    imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
    trainarray = np.array(imagebag.compute())  # PARALLELIZE
    trainarray = np.reshape(trainarray, (ims_per_class, -1))    
    labelarray = np.full((train.shape[0], 1), i)
    trainarray = np.concatenate((labelarray, trainarray), axis=1)
    train_grand.append(trainarray)
    
train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

del trainarray
del train
