# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:15:54 2020

@author: Brayan
"""
 

import cv2 
import numpy as np

nice = cv2.imread('Emojis/felicidad.jpeg')
bad = cv2.imread('Emojis/tristeza.jpeg')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cara = faceClassif
faceClassif1 = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
ojos = faceClassif1



cap = cv2.VideoCapture(0)
while True:
   _,frame = cap.read()
   
       
   nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
   caras = face_cascade.detectMultiScale(gray, 1.3, 5)
   ojos = eye_cascade.detectMultiScale(gray, 1.3, 5)
   auxFrame = frame.copy()
 

   
   for (ox,oy,ow,oh) in ojos:
        cv2.rectangle(frame, (ox,oy),(ox+ow,oy+oh),(0,255,0),2)
        rostro = auxFrame[oy:oy+oh,ox:ox+ow]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        image = nice
        nFrame = cv2.hconcat([frame,image]) 
        

   for (x,y,w,h) in caras:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        image = bad
        nFrame = cv2.hconcat([frame,image]) 
 
     
   
   #imagen = image     
   	
   cv2.imshow('nFrame',nFrame)
   k = cv2.waitKey(1) 
   if k==27:
         break

cap.release()        
cv2.destroyAllwindows()         
