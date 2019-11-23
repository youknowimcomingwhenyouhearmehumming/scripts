import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math
import ThorFunctions3 as TH

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
#img = cv2.imread('pica36.png')
cv2.destroyAllWindows()
#
#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic')
#img = cv2.imread('video_574.png' )

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/closeup')
img = cv2.imread('closeup282.png' )



cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure('orignal')
imgplot = plt.imshow(cimg)

def colourmask(img,colour):
#Different applications use different scales for HSV. 
#For example gimp uses H = 0-360, S = 0-100 and V = 0-100. 
#But OpenCV uses H: 0-179, S: 0-255, V: 0-255. Here i got a hue value of 22 in gimp.
#So I took half of it, 11, and defined range for that. ie (5,50,50) - (15,255,255).
#    masked_img=[]
    try:

        if colour == 'red':
            low_red1 = np.array([162, 60, 60])     # ([165, 60, 60])
            high_red1 = np.array([179, 130, 200])  #[179, 130, 200])
            low_red2 = np.array([0, 0, 70])      #[0, 135, 20])
            high_red2 = np.array([0, 0, 255])   #([11, 200, 255])
            low_red3 = np.array([0, 135, 100])      #[0, 135, 20])
            high_red3 = np.array([11, 255, 255])   #([11, 200, 255])
            



            
            red_mask1 = cv2.inRange(img_hsv, low_red1, high_red1)
            red_mask2 = cv2.inRange(img_hsv, low_red2, high_red2)
            red_mask3 = cv2.inRange(img_hsv, low_red3, high_red3)
#            masked_img1 = cv2.bitwise_and(img, img, mask=red_mask1)
#            masked_img2 = cv2.bitwise_and(img, img, mask=red_mask2)
#            masked_img = cv2.bitwise_or(masked_img1, masked_img2)            
#            return red_mask1

            return cv2.bitwise_or(red_mask1,red_mask3)


    except:
        return None

def PreProcessing(img,number_of_iterations):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    dilated = cv2.dilate(img, element, iterations=number_of_iterations)
    eroded = cv2.erode(element, element, iterations=number_of_iterations)
    blurred = cv2.GaussianBlur(dilated, (11, 11), 0)  

    return blurred


img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


plt.figure('img hsv')
imgplot = plt.imshow(img_hsv)



red_mask=TH.colourmask(img,'red')
plt.figure('red mask')
imgplot = plt.imshow(red_mask)

red_mask_pp=TH.PreProcessing(red_mask,number_of_iterations=1)
plt.figure('red mask pp')
imgplot = plt.imshow(red_mask_pp)
    


#
#img=TH.colourmask(img,'red')
#
#
##cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.figure('mask')
#imgplot = plt.imshow(img)


#blurred = cv2.medianBlur(img, 9)
#_filter = cv2.bilateralFilter(blurred, 5, 75, 75)
#adap_thresh = cv2.adaptiveThreshold(_filter,
#                                    255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV,
#                                    21, 0)
#
#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
#eroded = cv2.erode(img, element, iterations=3)
#dilated = cv2.dilate(eroded, element, iterations=10)

# blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.blobColor = 0
#params.minThreshold = 14
#params.maxThreshold = 25
params.minDistBetweenBlobs=100
params.filterByArea = True
params.minArea = 100
params.maxArea = 20000
params.filterByCircularity = True
params.minCircularity =.8 #89 for img200  1, #90 img300 1,89 img610 3, 86 img 1150 3,  
params.maxCircularity = 1
params.filterByConvexity = True
params.minConvexity = 0.8#98 for img200 1,98 for img300 1,97 img610 20, 98 img 1150 2,   
params.maxConvexity=1
params.filterByInertia = True
params.minInertiaRatio=0.6 #84 for img200 2, #77 for img300 5, 75 img610 15, 55  img 1150 15, 
params.maxInertiaRatio=1



det = cv2.SimpleBlobDetector_create(params)
keypts = det.detect(red_mask_pp)

#im_with_keypoints = cv2.drawKeypoints(dilated,
#                                      keypts,
#                                      np.array([]),
#                                      (0, 0, 255),
#                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

res = cv2.drawKeypoints(img,
                        keypts,
                        np.array([]),
                        (0, 255, 0 ),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
x_pos=[]
y_pos=[]
i = 0
for kp in keypts:
    print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
    i+=1
    print(i)
    cv2.rectangle(img,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)
    x_pos.append(int(np.round(kp.pt[0])))
    y_pos.append(int(np.round(kp.pt[1])))

#cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("RESULT", res)
#cv2.imshow("adap_thresh", adap_thresh)
#cv2.waitKey(0)



