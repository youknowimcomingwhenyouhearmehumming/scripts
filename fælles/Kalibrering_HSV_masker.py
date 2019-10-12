import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils


def colourmask(img,colour):
#Different applications use different scales for HSV. 
#For example gimp uses H = 0-360, S = 0-100 and V = 0-100. 
#But OpenCV uses H: 0-179, S: 0-255, V: 0-255. Here i got a hue value of 22 in gimp.
#So I took half of it, 11, and defined range for that. ie (5,50,50) - (15,255,255).
#    masked_img=[]
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if colour == 'red':
            low_red1 = np.array([174, 100, 20])
            high_red1 = np.array([179, 255, 255])
#            low_red2 = np.array([0, 100, 20])
#            high_red2 = np.array([0, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, low_red1, high_red1)
#            red_mask2 = cv2.inRange(img_hsv, low_red2, high_red2)
#            masked_img1 = cv2.bitwise_and(img, img, mask=red_mask1)
#            masked_img2 = cv2.bitwise_and(img, img, mask=red_mask2)
#            masked_img = cv2.bitwise_or(masked_img1, masked_img2)            
            return red_mask1
#            return cv2.bitwise_or(red_mask1,red_mask2)
        elif colour == 'green':
            low_green = np.array([38, 50, 50])
            high_green = np.array([75, 255, 255])
            green_mask = cv2.inRange(img_hsv, low_green, high_green)
#            masked_img = cv2.bitwise_and(img, img, mask=green_mask)
#            return masked_img
            return green_mask
        elif colour == 'blue':
            low_blue = np.array([94, 80, 2])
            high_blue = np.array([126, 255, 255])
            blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)
#            masked_img = cv2.bitwise_and(img, img, mask=blue_mask)
#            return masked_img
            return blue_mask
        
    except:
        return None

plt.close()
cv2.destroyAllWindows()

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic')
file_name_of_picture='video_24.png'   #'pica33.png'
#
#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic')
#file_name_of_picture='video_1181.png'   #'pica33.png'


img = cv2.imread(file_name_of_picture,1)


img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('original')
imgplot = plt.imshow(img_RGB)
blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
plt.figure('HSV')
#hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
imgplot = plt.imshow(hsv)

plt.figure('h')
imgplot = plt.imshow(hsv[:,:,0])
plt.figure('s')
imgplot = plt.imshow(hsv[:,:,1])
plt.figure('v')
imgplot = plt.imshow(hsv[:,:,2])


img_red_mask=colourmask(blurred,'red')
    
#cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure('all_pos')
imgplot = plt.imshow(img_red_mask)




#PICA 33 x 598
#circles_h=h_method(img,1,500,200,13,10,40)
#circles_s=s_method(img,1,500,190,13,10,40)
#circles_v=v_method(img,1,500,195,16,10,40)
#circles_gray=gray_method(file_name_of_picture,1,200,300,11,10,40)
#PICA 34  x=636
#circles_h=h_method(img,1,150,200,13,10,40)
#circles_s=s_method(img,1,150,190,13,10,40)
#circles_v=v_method(img,1,150,195,16,10,40)
#circles_gray=gray_method(file_name_of_picture,1,150,300,11,10,40)
#PICA 35 x=676
#circles_h=h_method(img,1,150,200,13,5,40)
#circles_s=s_method(img,1,150,190,13,5,40)
#circles_v=v_method(img,1,150,150,15,5,40)
#circles_gray=gray_method(file_name_of_picture,1,150,300,11,10,40)
#

#A=np.array([10,11,13])
#B=np.array([8,7,2])
#C=np.array([12,18,16])
#
#mask=(A[:]<=B+1)
#
#mask_multiple_criteria=[(A < B) & (A > C) & (A==5)]






"""
This whol part is for combining the x and y component into one number
"""
#"""
#This parts combines the x and y components into one number
#"""
#h_pos_1D=np.sqrt(np.power(circles_h[0,0,0:2], 2)+np.power(circles_h[0,0,0:2],2))
#s_pos_1D=np.sqrt(np.power(circles_s[0,0,0:2], 2)+np.power(circles_s[0,0,0:2],2))
#v_pos_1D=np.sqrt(np.power(circles_v[0,0,0:2], 2)+np.power(circles_v[0,0,0:2],2))
#gray_pos_1D=np.sqrt(np.power(circles_gray[0,0,0:2], 2)+np.power(circles_gray[0,0,0:2],2))
#"""
#Here it's checked wheter each individual position is too far away from the oldposition +/- the threshold of 
#the old position.
#"""
#all_pos=np.concatenate((h_pos_1D,s_pos_1D,v_pos_1D,gray_pos_1D), axis=None)
#all_pos_valid_oldpos=all_pos[(oldpos-thres_oldpos < all_pos) & (oldpos+thres_oldpos > all_pos)]  
#
#"""
#"""
#avg_pos_all=np.mean(all_pos_valid_oldpos)
#
#while(np.sum([(avg_pos_all-thres_avg_pos < all_pos_valid_oldpos) & (avg_pos_all+thres_avg_pos < all_pos_valid_oldpos)])>0): #While some of the positions are outside the mean +/- threshold
#    all_pos_valid_oldpos[(avg_pos_all-thres_avg_pos < all_pos_valid_oldpos) & (avg_pos_all+thres_avg_pos > all_pos_valid_oldpos)]
#
#
#avg_pos(all_pos_valid_oldpos)
