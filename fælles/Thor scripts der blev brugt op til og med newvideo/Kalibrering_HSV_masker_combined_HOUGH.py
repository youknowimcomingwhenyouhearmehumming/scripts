import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math
import ThorFunctions2 as TH

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

Hough_pos = cv2.HoughCircles(red_mask_pp,cv2.HOUGH_GRADIENT,dp=1,minDist=1000,param1=400,param2=13,minRadius=3,maxRadius=50) 
    
Hough_pos = np.uint16(np.around(Hough_pos))
for i in Hough_pos[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)

cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure('original with circels')
imgplot = plt.imshow(cimg)







    
