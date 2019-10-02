import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils



def h_method(img_original,dbValue_h,minDist_h,param1_h,param2_h,minRadius_h,maxRadius_h):
    img=img_original.copy()
    blurred = cv2.GaussianBlur(img, (11, 11), 0) #Delte this line and import the blurred image directely
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    h_part=hsv[:,:,0]
    #cv2.imshow('h',h_part)  #Shows the original h part image of the hsv format
    iterations=2
    h_part = cv2.erode(h_part, None, iterations=iterations)
    h_part = cv2.dilate(h_part, None, iterations=iterations)
    
    
    circles_h = cv2.HoughCircles(h_part,cv2.HOUGH_GRADIENT,dbValue_h,minDist=minDist_h,
    param1=param1_h,param2=param2_h,minRadius=minRadius_h,maxRadius=maxRadius_h) 
    
    circles_h = np.uint16(np.around(circles_h))
    for i in circles_h[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
    
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure('h part')
    imgplot = plt.imshow(cimg)
    
    return circles_h
    
def s_method(img_original,dbValue_s,minDist_s,param1_s,param2_s,minRadius_s,maxRadius_s):
    img=img_original.copy()
    blurred = cv2.GaussianBlur(img, (11, 11), 0) #Delte this line and import the blurred image directely
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s_part=hsv[:,:,1]
    #cv2.imshow('s',s_part)  #Shows the original h part image of the hsv format
    iterations=2
    s_part = cv2.erode(s_part, None, iterations=iterations)
    s_part = cv2.dilate(s_part, None, iterations=iterations)
    
    
    circles_s = cv2.HoughCircles(s_part,cv2.HOUGH_GRADIENT,dbValue_s,minDist=minDist_s,
    param1=param1_s,param2=param2_s,minRadius=minRadius_s,maxRadius=maxRadius_s) 
    
    circles_s = np.uint16(np.around(circles_s))
    for i in circles_s[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
    
    cimg= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure('s part')
    imgplot = plt.imshow(cimg)
    
    return circles_s

def v_method(img_original,dbValue_v,minDist_v,param1_v,param2_v,minRadius_v,maxRadius_v):
    img=img_original.copy()
    blurred = cv2.GaussianBlur(img, (11, 11), 0) #Delte this line and import the blurred image directely
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    v_part=hsv[:,:,2]
    #cv2.imshow('h',v_part)  #Shows the original h part image of the hsv format
    iterations=2
    v_part = cv2.erode(v_part, None, iterations=iterations)
    v_part = cv2.dilate(v_part, None, iterations=iterations)
    
    
    circles_v = cv2.HoughCircles(v_part,cv2.HOUGH_GRADIENT,dbValue_v,minDist=minDist_v,
    param1=param1_v,param2=param2_v,minRadius=minRadius_v,maxRadius=maxRadius_v) 
    
    circles_v = np.uint16(np.around(circles_v))
    print(circles_v)
    for i in circles_v[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
    
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure('v part')
    imgplot = plt.imshow(cimg)
    
    return circles_v


def gray_method(file_name_of_image,dbValue_gray,minDist_gray,param1_gray,param2_gray,minRadius_gray,maxRadius_gray):
    img=cv2.imread(file_name_of_image,0)   #The file name is needed and not just the file since the file have to be read in two different ways:
    img_gray_show= cv2.imread(file_name_of_image,1)
    
    blurred = cv2.GaussianBlur(img, (11, 11), 0) #Delte this line and import the blurred image directely
    #cv2.imshow('gray',img) #Shows the original gray picture
    
    iterations=2
    gray_part = cv2.erode(blurred, None, iterations=iterations)
    gray_part = cv2.dilate(blurred, None, iterations=iterations)
    
    
    circles_gray = cv2.HoughCircles(gray_part,cv2.HOUGH_GRADIENT,dbValue_gray,minDist=minDist_gray,
    param1=param1_gray,param2=param2_gray,minRadius=minRadius_gray,maxRadius=maxRadius_gray) 
    
    circles_gray = np.uint16(np.around(circles_gray))
    for i in circles_gray[0,:]:
        # draw the outer circle
        cv2.circle(img_gray_show,(i[0],i[1]),i[2],(0,255,0),12)
        # draw the center of the circle
        cv2.circle(img_gray_show,(i[0],i[1]),2,(0,0,255),8)
    
    cimg_gray = cv2.cvtColor(img_gray_show,cv2.COLOR_BGR2RGB)
    plt.figure('gray')
    imgplot = plt.imshow(cimg_gray)

    return circles_gray

        



os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
plt.close()

file_name_of_picture='pica36.png'   #'pica33.png'
img = cv2.imread(file_name_of_picture,1)

img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('original')
imgplot = plt.imshow(img_RGB)
blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
plt.figure('HSV')
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
imgplot = plt.imshow(hsv_rgb)



circles_h=h_method(img,1,150,130,14,5,40)
circles_s=s_method(img,1,150,130,15,5,40)
circles_v=v_method(img,1,150,150,15,5,40)
circles_gray=gray_method(file_name_of_picture,1,150,300,11,10,40)


#circles_h_pos=circles_h[0,0,0:2]
#circles_s_pos=circles_s[0,0,0:2]
#circles_v_pos=circles_v[0,0,0:2]
#circles_gray_pos=circles_gray[0,0,0:2]


"""
Here the positon is calculated for the x-part
"""
oldpos_x=676
thres_oldpos=50
thres_avg_pos=10

"""
Here it's checked wheter each individual position is too far away from the oldposition +/- the threshold of 
the old position.
"""
all_pos_x=np.array((0))
try: 
    all_pos_x=np.concatenate((all_pos_x,circles_h[0,:,0]), axis=None)
except:
    pass
try: 
    all_pos_x=np.concatenate((all_pos_x,circles_s[0,:,0]), axis=None)
except:
    pass
try: 
    all_pos_x=np.concatenate((all_pos_x,circles_v[0,:,0]), axis=None)
except:
    pass
try: 
    all_pos_x=np.concatenate((all_pos_x,circles_gray[0,:,0]), axis=None)
except:
    pass
all_pos_x = np.delete(all_pos_x, 0)

#all_pos_x=np.concatenate((all_pos_x,circles_h[0,:,0],circles_s[0,:,0],circles_v[0,:,0],circles_gray[0,:,0]), axis=None)
all_pos_valid_oldpos_x=all_pos_x[(oldpos_x-thres_oldpos < all_pos_x) & (oldpos_x+thres_oldpos > all_pos_x)]  

"""
Here the x-positions that are more than thres_avg_pos away from the average is discarded.
The proces is repeated so a new average is calculated until noting is sorted away anymore.
"""

selected_pos_valid_x=all_pos_valid_oldpos_x
go=True
while(go): #While some of the positions are outside the mean +/- threshold
    avg_pos_x=np.mean(all_pos_valid_oldpos_x)
    selected_pos_valid_x=selected_pos_valid_x[(avg_pos_x-thres_avg_pos < selected_pos_valid_x) & (avg_pos_x+thres_avg_pos > selected_pos_valid_x)]
    number_of_elements_outside_threshold=np.sum([(avg_pos_x-thres_avg_pos < selected_pos_valid_x) & (avg_pos_x+thres_avg_pos < selected_pos_valid_x)])
    print('while')
    if number_of_elements_outside_threshold==0:
        go=False
        
avg_pos_final=np.mean(selected_pos_valid_x)
print('avgpos_x',avg_pos_final)
"""
retuner den endelige position
lav et plot på bilelde med alle de muligede position og så avg_pos_final
"""

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
#avg_pos_final(all_pos_valid_oldpos)
