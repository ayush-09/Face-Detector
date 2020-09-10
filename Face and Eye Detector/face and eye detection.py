# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:45:00 2020

@author: ayush
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r'A:\OpenCv\baby.jpg')
plt.imshow(img)
fix_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
eye_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faces_rects=face_classifier.detectMultiScale(fix_img,1.3,5)
def detect_face(fix_img):
    faces_rects=face_classifier.detectMultiScale(fix_img)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(fix_img,
                      (x,y),
                      (x+w,y+h),
                      (0,255,0),
                      5)
    return fix_img
def detect_eyes(fix_img):
    eye_rects=eye_classifier.detectMultiScale(fix_img)
    for (x,y,w,h) in eye_rects:
        cv2.rectangle(fix_img,
                      (x,y),
                      (x+w,y+h),
                      (255,255,255),
                      10)
    return fix_img
results=detect_face(fix_img)
results= detect_eyes(fix_img)
plt.imshow(results)
