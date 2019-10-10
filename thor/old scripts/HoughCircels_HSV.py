import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import imutils


os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
plt.close()

name_of_picture='pica33.png'
img = cv2.imread(name_of_picture,1)
img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



img_h = cv2.imread(name_of_picture,1)
img_s = cv2.imread(name_of_picture,1)
img_gray = cv2.imread(name_of_picture,0)
img_gray_show= cv2.imread(name_of_picture,1)



#greenLower = (160, 80, 30)
#greenUpper = (185, 255, 150)
#  
# resize the frame, blur it, and convert it to the HSV
# color space
#frame = imutils.resize(img, width=600)
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


dbValue_gray=1
minDist_gray=100
param1_gray=500
param2_gray=11
minRadius_gray=10
maxRadius_gray=40




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



#imgplot = plt.imshow(hsv)
#imgplot = plt.imshow(h_part)
#imgplot = plt.imshow(s_part)
#imgplot = plt.imshow(v_part)

#h_part = cv2.cvtColor(h_part,cv2.COLOR_GRAY2BGR)

#circles_h = cv2.HoughCircles(h_part,cv2.HOUGH_GRADIENT,1,minDist=50,
#param1=50,param2=5,minRadius=5,maxRadius=40) 
"""
S part
"""
#circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=200,param2=13,minRadius=10,maxRadius=40) 
circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,dbValue_s,minDist=minDist_s,
param1=param1_s,param2=param2_s,minRadius=minRadius_s,maxRadius=maxRadius_s) 
#For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
#pica 33: circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=200,param2=13,minRadius=10,maxRadius=40) 

circles_s = np.uint16(np.around(circles_s))
for i in circles_s[0,:]:
    # draw the outer circle
    cv2.circle(img_s,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_s,(i[0],i[1]),2,(0,0,255),8)
#cv2.imshow('detected circles_h',cimg)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imshow('s part',img_s)
cimg_s = cv2.cvtColor(img_s,cv2.COLOR_BGR2RGB)
plt.figure('s part')
imgplot = plt.imshow(cimg_s)

"""
H part
"""
circles_h = cv2.HoughCircles(h_part,cv2.HOUGH_GRADIENT,dbValue_h,minDist=minDist_h,
param1=param1_h,param2=param2_h,minRadius=minRadius_h,maxRadius=maxRadius_h) 

#For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
#pica 33: circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=200,param2=13,minRadius=10,maxRadius=40) 

circles_h = np.uint16(np.around(circles_h))
for i in circles_h[0,:]:
    # draw the outer circle
    cv2.circle(img_h,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_h,(i[0],i[1]),2,(0,0,255),8)
#cv2.imshow('detected circles_h',cimg)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imshow('h part',img_h)

cimg_h = cv2.cvtColor(img_h,cv2.COLOR_BGR2RGB)
plt.figure('h part')
imgplot = plt.imshow(cimg_h)



"""
RGB to gray
"""
circles_gray = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,dbValue_gray,minDist=minDist_gray,
param1=param1_gray,param2=param2_gray,minRadius=minRadius_gray,maxRadius=maxRadius_gray) 


#For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
#pica 33: circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,1,minDist=500,
#param1=200,param2=13,minRadius=10,maxRadius=40) 

circles_gray = np.uint16(np.around(circles_gray))
for i in circles_gray[0,:]:
    # draw the outer circle
    cv2.circle(img_gray_show,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img_gray_show,(i[0],i[1]),2,(0,0,255),8)
#cv2.imshow('detected circles_gray',cimg)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imshow('h part',img_gray)

cimg_gray = cv2.cvtColor(img_gray_show,cv2.COLOR_BGR2RGB)
plt.figure('gray')
imgplot = plt.imshow(cimg_gray)






#
### construct a mask for the color "green", then perform
### a series of dilations and erosions to remove any small
### blobs left in the mask
##mask = cv2.inRange(hsv, greenLower, greenUpper)
##mask = cv2.erode(mask, None, iterations=2)
##mask = cv2.dilate(mask, None, iterations=2)
##
##cv2.imshow('e',mask)
#
#
## find contours in the mask and initialize the current
## (x, y) center of the ball
#cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#center = None
#
## only proceed if at least one contour was found
#if len(cnts) > 0:
#	# find the largest contour in the mask, then use
#	# it to compute the minimum enclosing circle and
#	# centroid
#	c = max(cnts, key=cv2.contourArea)
#	((x, y), radius) = cv2.minEnclosingCircle(c)
#	M = cv2.moments(c)
#	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#
#	# only proceed if the radius meets a minimum size
#	if radius > 10:
#		# draw the circle and centroid on the frame,
#		# then update the list of tracked points
#		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
#		cv2.circle(frame, center, 5, (0, 0, 255), -1)
#        
#
#cv2.imshow('r',frame)
#
#
#
#
##img = cv2.imread('pica55.png',0)
##img = cv2.medianBlur(img,5)
##cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
###cimg = cv2.erode(cimg, None, iterations=1)
###cimg = cv2.dilate(cimg, None, iterations=1)
##
##circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=500,
##param1=500,param2=12,minRadius=5,maxRadius=40) 
###For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
###For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
## 
##circles = np.uint16(np.around(circles))
##for i in circles[0,:]:
##    # draw the outer circle
##    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
##    # draw the center of the circle
##    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
###cv2.imshow('detected circles',cimg)
##    
##plt.figure()
##imgplot = plt.imshow(cimg)
##
###cv2.waitKey(0)
###cv2.destroyAllWindows()
#
#
#
