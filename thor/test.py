import cv
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')

plt.close()
cv2.destroyAllWindows()
img = cv.LoadImage('pica51.png',1)
bluered = cv2.medianBlur(img,5)
#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cimg_hsv = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
hsv_bluered = cv2.cvtColor(bluered, cv2.COLOR_BGR2HSV)



cv2.imshow('img',img)
#plt.figure(2)
#imgplot = plt.imshow(cimg)
cv2.imshow('hsv',hsv)
cv2.imshow('bluered',bluered)
cv2.imshow('sv_bluered',hsv_bluered)
plt.figure(1)
imgplot = plt.imshow(hsv)
#plt.colors.rgb_to_hsv(arr)


#plt.figure()
#imgplot = plt.imshow(cimg_hsv)
#
##cimg = cv2.erode(cimg, None, iterations=1)
##cimg = cv2.dilate(cimg, None, iterations=1)
#
#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=500,param2=12,minRadius=5,maxRadius=40) 
##For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
##For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
# 
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
##cv2.imshow('detected circles',cimg)
#    
#plt.figure()
#imgplot = plt.imshow(cimg)
##
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#
#
#
#
