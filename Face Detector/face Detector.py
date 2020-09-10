# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:13:53 2020

@author: ayush
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
image=cv2.imread(r'A:\OpenCv\baby.jpg')
fix_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
faces_rects=face_classifier.detectMultiScale(image,1.3,5)
if faces_rects is ():
    print('No Faces found')
def detect_face(fix_img):
    faces_rects=face_classifier.detectMultiScale(fix_img)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(fix_img,
                      (x,y),
                      (x+w,y+h),
                      (0,255,0),
                      5)
    return fix_img
results=detect_face(image)
plt.imshow(results)
