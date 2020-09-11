# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:08:08 2020

@author: dell 1
"""


#import all the library needed for training
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate


#import training data from the folders that contains all the images for training:
path = Path('/CCAS/home/jialu/coursev3/nbs/dl1/divercity/') 
##path of the folder that contains the falder of training images
data3 = (ImageDataBunch.from_csv(path, size=200, folder='allimages',
                                #label_delim=' ', 
                                bs=16, valid_pct=0.2, ds_tfms=get_transforms(do_flip=False))
                                .normalize(imagenet_stats))
##transform the data in to package-required format: pixelsize "size"; specify the which the folder all the images are in, defien batch size, the smaller the quicker for training, usually 16 or 32; split data into training set and validation set, "valid_pct" define the proportion data to validation set; transform the data by "ds_tfms", which is not  must.


#start traning model:

##step1: train added layers
learn_bt = cnn_learner(data3, models.resnet34, metrics=error_rate)
learn_bt.fit_one_cycle(5)
learn_bt.unfreeze()

##step2: find suitable leanring rate through the plor curve. Choose the inteval right before the currve start to increase.
learn_bt.lr_find()
learn_bt.recorder.plot()

##step3:unfreeze all the layers to train. The number of layers can be changed untill the error rate not decrease any more. The comparision of training loss and validation losee doesn/t matter.
learn_bt.fit_one_cycle(12, lrs)
#Here we have our final model.


#import all the library for puutting model into production
import cv2
import face_recognition
import glob
import numpy as np
import PIL
from PIL import Image as PIL_Image



#import the images for prediction from a folder
types = ('*.jpeg', '*.jpg','*.png') # the tuple of image file types
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob('/CCAS/home/jialu/coursev3/nbs/dl1/divercity/testimages/*'+ files))

    
#Crop faces from oriinal images:
##grab the new images and put it into a link, and creast the list of location of faces in each picture
Image_list=[]
Image_list_face_locations=[]
for files_grabbed_path in files_grabbed:
    image = face_recognition.load_image_file(files_grabbed_path)
    face_locations = face_recognition.face_locations(image)
    Image_list.append(image)
    Image_list_face_locations.append(face_locations)

##get the locations of each face
toplist=[]
rightlist=[]
bottomlist=[]
leftlist=[]
for face_location in Image_list_face_locations:
    toplist.append(face_location[0][0])
    rightlist.append(face_location[0][1])
    bottomlist.append(face_location[0][2])
    leftlist.append(face_location[0][3])
    
##get the cropped face of each picture    
CroppedFace=[]
for i in range(len(Image_list)):
    # You can access the actual face itself like this:
    face_image = Image_list[i][toplist[i]:bottomlist[i], leftlist[i]:rightlist[i]]
    pil_image = PIL_Image.fromarray(face_image) #type(pil_image) is PIL.Image.Image
    CroppedFace.append(pil_image)

#Put Model into Production
## Get the predictions of all images at once from a folder and save it into list "prediction"
prediction=[]
for i in range(29):
    im=open_image(glob.glob('/CCAS/home/jialu/coursev3/nbs/dl1/divercity/testimages_CroppedFace/*.jpeg')[i])
    pred_class,pred_idx,outputs = learn_bt.predict(im)
    prediction.append(pred_class.obj)

    
##save the train model, export the weights of the CNN and reload it into other consoles 
learn_bt.export(file=Path("/CCAS/home/jialu/coursev3/nbs/dl1/divercity/export.pkl")) 
# my model weights are saved in the export.pkl
disployed_path="/CCAS/home/jialu/coursev3/nbs/dl1/divercity/"
learn_try=load_learner(disployed_path)