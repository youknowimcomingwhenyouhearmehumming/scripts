import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math


os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')

print('start')

b=3

# Read image
im = cv2.imread('pica36.png')

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic')
im = cv2.imread('video_574.png' )


imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
# get contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_area = []
# calculate area and filter into new array
for con in contours:
    area = cv2.contourArea(con)
    if 10 < area < 1000000:
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
        
        
cv2.drawContours(im, contours_cirles, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', im) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
