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
img = cv2.imread('pica36.png')
# convert image to grayscale image


#img = cv2.imread('blob.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)
_filter = cv2.bilateralFilter(blurred, 5, 75, 75)
adap_thresh = cv2.adaptiveThreshold(_filter,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    21, 0)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
dilated = cv2.dilate(adap_thresh, element, iterations=1)

# blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.minThreshold = 75
params.maxThreshold = 86
params.blobColor = 0
params.minDistBetweenBlobs=100
params.filterByArea = True
params.minArea = 150
params.maxArea = 300
params.filterByCircularity = True
params.minCircularity =.2
params.maxCircularity = 1
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity=0.7


det = cv2.SimpleBlobDetector_create(params)
keypts = det.detect(dilated)

im_with_keypoints = cv2.drawKeypoints(dilated,
                                      keypts,
                                      np.array([]),
                                      (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

res = cv2.drawKeypoints(img,
                        keypts,
                        np.array([]),
                        (0, 0, 255 ),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

i = 0
for kp in keypts:
    print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
    i+=1
    cv2.rectangle(res,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)

#cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("RES", res)
cv2.imshow("adap_thresh", adap_thresh)
#cv2.waitKey(0)


#import inspect
#lines = inspect.getsource(cv2.SimpleBlobDetector_create())
#print(lines)


inspect.getsourcefile(cv2.SimpleBlobDetector_create())

print(inspect.getsource(type(cv2.SimpleBlobDetector_create)))
