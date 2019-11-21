import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import math
import ThorFunctions3 as TH



cv2.destroyAllWindows()
plt.close('all')
#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic')
#img = cv2.imread('video_1222.png' )


#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/closeup')
#img = cv2.imread('closeup1200.png' )

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/longAndShortDist')
#img = cv2.imread('longAndShortDist450.png' )

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/hd')
img = cv2.imread('hd560.png' )


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
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_component=img_hsv[:,:,0]
        s_component=img_hsv[:,:,1]
        if colour == 'red':
            low_red1 = np.array([162, 60, 60])     # ([165, 60, 60])
            high_red1 = np.array([179, 130, 200])  #[179, 130, 200])
            low_red2 = np.array([0, 0, 70])      #[0, 135, 20])
            high_red2 = np.array([0, 0, 255])   #([11, 200, 255])
            low_red3 = np.array([0, 135, 100])      #[0, 135, 20])
            high_red3 = np.array([11, 255, 255])   #([11, 200, 255])
            
            
            mask_far_dist1=h_component==0
            mask_far_dist2=s_component==0
            mask_far_dist=(mask_far_dist1*mask_far_dist2)*1

            
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



red_mask=colourmask(img,'red')
plt.figure('red mask')
imgplot = plt.imshow(red_mask)

red_mask_pp=PreProcessing(red_mask,number_of_iterations=1)
plt.figure('red mask pp')
imgplot = plt.imshow(red_mask_pp)
    




findContour_pos=TH.findContours(red_mask_pp)
    


#im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
ret, thresh = cv2.threshold(red_mask_pp, 1, 255, 0)
#contours,hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#New lines added from pyimage
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
c = max(contours, key=cv2.contourArea)


    	

#This is not used since we only want to find the centrum and radius after sorting on area and circulatiry
#cnt = contours[0]
#M = cv2.moments(cnt)
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#        
#(x,y),radius = cv2.minEnclosingCircle(cnt)
#center_new = (int(x),int(y))
#radius_new  = int(radius)


#Next part is discarding based on area and circularity
contours_area = []
# calculate area and filter into new array
for con in contours:
    area = cv2.contourArea(con)
    print('area=',area)
    if 300 < area < 40000:
        contours_area.append(con)
        
        
contours_cirles = []

# check if contour is of circular shape
for con in contours_area:
    perimeter = cv2.arcLength(con, True)
    area = cv2.contourArea(con)
    if perimeter == 0:
        break
    circularity = 4*math.pi*(area/(perimeter*perimeter))
    print ('circularity=',circularity)
    if 0.72 < circularity < 1.2:
        contours_cirles.append(con)
        


        
#Next part: finding minimum enclosing circel
cnt = contours_cirles[0]
(x,y),radius = cv2.minEnclosingCircle(cnt)
center_new = (int(x),int(y))
radius_new  = int(radius)


findContour_pos= np.hstack((x,y,radius) )
findContour_pos=np.reshape(findContour_pos, (1,1, 3))



#    #The next par is only to find the extremums in N,E,S,W and thereafter drawing these and the perimeter itself
# determine the most extreme points along the contour
#    extLeft = tuple(c[c[:, :, 0].argmin()][0])
#    extRight = tuple(c[c[:, :, 0].argmax()][0])
#    extTop = tuple(c[c[:, :, 1].argmin()][0])
#    extBot = tuple(c[c[:, :, 1].argmax()][0])
#    
#    diameter1=TH.calculateDistance(extLeft,extRight)
#    diameter2=TH.calculateDistance(extTop,extBot)
#    
#    if diameter1 >= diameter2:
#        radius=np.round(diameter1/2)
#    else:
#        radius=np.round(diameter2/2)
#    
#    #Only if you want to draw the extrema
#    cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
#    cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
#    cv2.circle(img, extRight, 8, (0, 255, 0), -1)
#    cv2.circle(img, extTop, 8, (255, 0, 0), -1)
#    cv2.circle(img, extBot, 8, (255, 255, 0), -1)

findContour_pos = np.uint16(np.around(findContour_pos))
for i in findContour_pos[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)


cv2.imshow('Contours', img) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
