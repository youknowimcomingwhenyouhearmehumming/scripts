import cv2
import imutils
import os

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/ballpictures_jesper/TestDrone640_2')



cv2.namedWindow('w')
cam0=cv2.VideoCapture('image_98.png')
i=0
(ret,img)=cam0.read()
cv2.imshow('w',img)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#img=mpimg.imread('image_53.png')
imgplot = plt.imshow(img)


greenLower = (160, 80, 30)
greenUpper = (185, 255, 150)
  
# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize(img, width=600)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#img=mpimg.imread('image_53.png')
imgplot = plt.imshow(hsv)

# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

cv2.imshow('e',mask)


# find contours in the mask and initialize the current
# (x, y) center of the ball
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None

# only proceed if at least one contour was found
if len(cnts) > 0:
	# find the largest contour in the mask, then use
	# it to compute the minimum enclosing circle and
	# centroid
	c = max(cnts, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(c)
	M = cv2.moments(c)
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	# only proceed if the radius meets a minimum size
	if radius > 10:
		# draw the circle and centroid on the frame,
		# then update the list of tracked points
		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
        

cv2.imshow('r',frame)
