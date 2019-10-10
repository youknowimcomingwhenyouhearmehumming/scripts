import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math


#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
#img = cv2.imread('pica36.png')


os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic')
img = cv2.imread('video_610.png' )

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
params.blobColor = 0
#params.minThreshold = 14
#params.maxThreshold = 25
params.minDistBetweenBlobs=1
params.filterByArea = True
params.minArea = 5
params.maxArea = 100000
params.filterByCircularity = True
params.minCircularity =.84 #89 for img200  1, #90 img300 1,89 img610 3, 86 img 1150 3,  
params.maxCircularity = 1
params.filterByConvexity = True
params.minConvexity = 0.96#98 for img200 1,98 for img300 1,97 img610 20, 98 img 1150 2,   
params.maxConvexity=1
params.filterByInertia = True
params.minInertiaRatio=0.50 #84 for img200 2, #77 for img300 5, 75 img610 15, 55  img 1150 15, 
params.maxInertiaRatio=1



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
x_pos=[]
y_pos=[]
i = 0
for kp in keypts:
    print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
    i+=1
    cv2.rectangle(res,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)
    x_pos.append(int(np.round(kp.pt[0])))
    y_pos.append(int(np.round(kp.pt[1])))

#cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("RES", res)
cv2.imshow("adap_thresh", adap_thresh)
#cv2.waitKey(0)



