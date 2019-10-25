import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math
import ThorFunctions2 as TH


#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
#
#print('start')
#
#b=3
#
## Read image
#im = cv2.imread('pica36.png')
#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
#img = cv2.imread('pica36.png')
cv2.destroyAllWindows()
os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic')
img = cv2.imread('video_574.png' )




cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure('orignal')
imgplot = plt.imshow(cimg)


red_mask=TH.colourmask(img,'red')
plt.figure('red mask')
imgplot = plt.imshow(red_mask)

red_mask_pp=TH.PreProcessing(red_mask)
plt.figure('red mask pp')
imgplot = plt.imshow(red_mask_pp)



#im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
ret, thresh = cv2.threshold(red_mask_pp, 1, 255, 0)
#contours,hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#New lines added from pyimage
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
c = max(contours, key=cv2.contourArea)


	
# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

diameter1=TH.calculateDistance(extLeft,extRight)
diameter2=TH.calculateDistance(extTop,extBot)

cnt = contours[0]
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
        
(x,y),radius = cv2.minEnclosingCircle(cnt)
center_new = (int(x),int(y))
radius_new  = int(radius)



if diameter1 >= diameter2:
    radius=np.round(diameter1/2)
else:
    radius=np.round(diameter2/2)

#Only if you want to draw the extrema
cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
cv2.circle(img, extRight, 8, (0, 255, 0), -1)
cv2.circle(img, extTop, 8, (255, 0, 0), -1)
cv2.circle(img, extBot, 8, (255, 255, 0), -1)
 


contours_area = []
# calculate area and filter into new array
for con in contours:
    area = cv2.contourArea(con)
    print(area)
    if 100 < area < 40000:
        contours_area.append(con)
        
        
contours_cirles = []

# check if contour is of circular shape
for con in contours_area:
    perimeter = cv2.arcLength(con, True)
    area = cv2.contourArea(con)
    if perimeter == 0:
        break
    circularity = 4*math.pi*(area/(perimeter*perimeter))
    print (circularity)
    if 0.1 < circularity < 1.2:
        contours_cirles.append(con)
        


        
cv2.drawContours(img, contours_cirles, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', img) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
