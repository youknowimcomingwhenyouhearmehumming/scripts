import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import imutils


os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
plt.close()

name_of_picture='pica33.png'   #'pica33.png'
img = cv2.imread(name_of_picture,1)
img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



img_h = cv2.imread(name_of_picture,1)
img_s = cv2.imread(name_of_picture,1)
img_v = cv2.imread(name_of_picture,1)
img_gray = cv2.imread(name_of_picture,0)
img_gray_show= cv2.imread(name_of_picture,1)


blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

h_part=hsv[:,:,0]
s_part=hsv[:,:,1]
v_part=hsv[:,:,2]

plt.figure('original')
imgplot = plt.imshow(img_RGB)

cv2.imshow('h',h_part)
cv2.imshow('s',s_part)
cv2.imshow('v',v_part)
cv2.imshow('gray',img_gray)

cv2.imshow('hsv',hsv)

plt.figure('HSV')
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
imgplot = plt.imshow(hsv_rgb)



dbValue_s=1
minDist_s=500
param1_s=200
param2_s=13
minRadius_s=10
maxRadius_s=40

dbValue_h=1
minDist_h=500
param1_h=200
param2_h=13
minRadius_h=10
maxRadius_h=40

dbValue_v=1
minDist_v=50
param1_v=300
param2_v=15
minRadius_v=10
maxRadius_v=40

dbValue_gray=1
minDist_gray=100
param1_gray=500
param2_gray=11
minRadius_gray=10
maxRadius_gray=40

"""
Earlier used values
"""
#For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
#pica 33: circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=200,param2=13,minRadius=10,maxRadius=40) 



"""
dialate and erode
"""
iterations=2
h_part = cv2.erode(h_part, None, iterations=iterations)
h_part = cv2.dilate(h_part, None, iterations=iterations)
s_part = cv2.erode(s_part, None, iterations=iterations)
s_part = cv2.dilate(s_part, None, iterations=iterations)
v_part = cv2.erode(v_part, None, iterations=iterations)
v_part = cv2.dilate(v_part, None, iterations=iterations)




"""
S part
"""
circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,dbValue_s,minDist=minDist_s,
param1=param1_s,param2=param2_s,minRadius=minRadius_s,maxRadius=maxRadius_s) 

circles_s = np.uint16(np.around(circles_s))
for i in circles_s[0,:]:
    # draw the outer circle
    cv2.circle(img_s,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_s,(i[0],i[1]),2,(0,0,255),8)

cimg_s = cv2.cvtColor(img_s,cv2.COLOR_BGR2RGB)
plt.figure('s part')
imgplot = plt.imshow(cimg_s)

"""
H part
"""
circles_h = cv2.HoughCircles(h_part,cv2.HOUGH_GRADIENT,dbValue_h,minDist=minDist_h,
param1=param1_h,param2=param2_h,minRadius=minRadius_h,maxRadius=maxRadius_h) 

circles_h = np.uint16(np.around(circles_h))
for i in circles_h[0,:]:
    # draw the outer circle
    cv2.circle(img_h,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_h,(i[0],i[1]),2,(0,0,255),8)

cimg_h = cv2.cvtColor(img_h,cv2.COLOR_BGR2RGB)
plt.figure('h part')
imgplot = plt.imshow(cimg_h)


"""
V part
"""
circles_v = cv2.HoughCircles(v_part,cv2.HOUGH_GRADIENT,dbValue_v,minDist=minDist_v,
param1=param1_v,param2=param2_v,minRadius=minRadius_v,maxRadius=maxRadius_v) 

circles_v = np.uint16(np.around(circles_v))
for i in circles_v[0,:]:
    # draw the outer circle
    cv2.circle(img_v,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_v,(i[0],i[1]),2,(0,0,255),8)

cimg_v = cv2.cvtColor(img_v,cv2.COLOR_BGR2RGB)
plt.figure('v part')
imgplot = plt.imshow(cimg_v)



"""
RGB to gray
"""
circles_gray = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,dbValue_gray,minDist=minDist_gray,
param1=param1_gray,param2=param2_gray,minRadius=minRadius_gray,maxRadius=maxRadius_gray) 

circles_gray = np.uint16(np.around(circles_gray))
for i in circles_gray[0,:]:
    # draw the outer circle
    cv2.circle(img_gray_show,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_gray_show,(i[0],i[1]),2,(0,0,255),8)

cimg_gray = cv2.cvtColor(img_gray_show,cv2.COLOR_BGR2RGB)
plt.figure('gray')
imgplot = plt.imshow(cimg_gray)



