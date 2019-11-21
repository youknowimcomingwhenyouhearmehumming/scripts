import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils



def h_method(img_original,dbValue_h,minDist_h,param1_h,param2_h,minRadius_h,maxRadius_h):
    try:    
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
        
#        print('Succed')
        return circles_h

    except:
#        print('not succed')
        pass
    
    
def s_method(img_original,dbValue_s,minDist_s,param1_s,param2_s,minRadius_s,maxRadius_s):
    try:
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

    except:
        pass

def v_method(img_original,dbValue_v,minDist_v,param1_v,param2_v,minRadius_v,maxRadius_v):
    try:
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
        for i in circles_v[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
        
        cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.figure('v part')
        imgplot = plt.imshow(cimg)
        
        return circles_v

    except:
        pass


def gray_method(file_name_of_image,dbValue_gray,minDist_gray,param1_gray,param2_gray,minRadius_gray,maxRadius_gray):
    try:
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


    except:
        pass

        


def concatenate_results(method1,method2,method3,method4):
    
    all_pos_x=np.array((0))
    try: 
        all_pos_x=np.concatenate((all_pos_x,method1[0,:,0]), axis=None)
    except:
        pass
    try: 
        all_pos_x=np.concatenate((all_pos_x,method2[0,:,0]), axis=None)
    except:
        pass
    try: 
        all_pos_x=np.concatenate((all_pos_x,method3[0,:,0]), axis=None)
    except:
        pass
    try: 
        all_pos_x=np.concatenate((all_pos_x,method4[0,:,0]), axis=None)
    except:
        pass
    all_pos_x = np.delete(all_pos_x, 0)
    
    
    all_pos_y=np.array((0))
    try: 
        all_pos_y=np.concatenate((all_pos_y,method1[0,:,1]), axis=None)
    except:
        pass
    try: 
        all_pos_y=np.concatenate((all_pos_y,method2[0,:,1]), axis=None)
    except:
        pass
    try: 
        all_pos_y=np.concatenate((all_pos_y,method3[0,:,1]), axis=None)
    except:
        pass
    try: 
        all_pos_y=np.concatenate((all_pos_y,method4[0,:,1]), axis=None)
    except:
        pass
    all_pos_y = np.delete(all_pos_y, 0)
    
    
    all_radius=np.array((0))
    try: 
        all_radius=np.concatenate((all_radius,method1[0,:,2]), axis=None)
    except:
        pass
    try: 
        all_radius=np.concatenate((all_radius,method2[0,:,2]), axis=None)
    except:
        pass
    try: 
        all_radius=np.concatenate((all_radius,method3[0,:,2]), axis=None)
    except:
        pass
    try: 
        all_radius=np.concatenate((all_radius,method4[0,:,2]), axis=None)
    except:
        pass
    all_radius = np.delete(all_radius, 0)
    
    
    return all_pos_x,all_pos_y,all_radius
    

def Discard_if_too_far_from_old_pos(all_pos,oldpos,thres_oldpos):
    """
    Here it's checked wheter each individual position is too far away from the oldposition +/- the threshold of 
    the old position.
    """
    all_pos_valid_oldpos=all_pos[(oldpos-thres_oldpos < all_pos) & (oldpos+thres_oldpos > all_pos)]  
    return all_pos_valid_oldpos



def Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos,thres_avg_pos):
    """
    Here the x-positions that are more than thres_avg_pos away from the average is discarded.
    The proces is repeated so a new average is calculated until noting is sorted away anymore.
    """
    selected_pos_valid=all_pos_valid_oldpos
    go=True
    while(go): #While some of the positions are outside the mean +/- threshold
        avg_pos=np.mean(all_pos_valid_oldpos)
        selected_pos_valid=selected_pos_valid[(avg_pos-thres_avg_pos < selected_pos_valid) & (avg_pos+thres_avg_pos > selected_pos_valid)]
        number_of_elements_outside_threshold=np.sum([(avg_pos-thres_avg_pos < selected_pos_valid) & (avg_pos+thres_avg_pos < selected_pos_valid)])
#        print('while')
        if number_of_elements_outside_threshold==0:
            go=False
            
    avg_pos=np.mean(selected_pos_valid)
    return np.round(avg_pos)


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

def BLOB(img):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
    eroded = cv2.erode(img, element, iterations=3)
    dilated = cv2.dilate(eroded, element, iterations=10)
    
    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.blobColor = 0
    #params.minThreshold = 14
    #params.maxThreshold = 25
    params.minDistBetweenBlobs=10
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity =.7 #89 for img200  1, #90 img300 1,89 img610 3, 86 img 1150 3,  
    params.maxCircularity = 1
    params.filterByConvexity = True
    params.minConvexity = 0.90#98 for img200 1,98 for img300 1,97 img610 20, 98 img 1150 2,   
    params.maxConvexity=1
    params.filterByInertia = True
    params.minInertiaRatio=0.6 #84 for img200 2, #77 for img300 5, 75 img610 15, 55  img 1150 15, 
    params.maxInertiaRatio=1
                    
    
    
    det = cv2.SimpleBlobDetector_create(params)
    keypts = det.detect(dilated)
    
    all_pos_x=[]
    all_pos_y=[]
#        i = 0
    for kp in keypts:

#            i+=1
#        cv2.rectangle(res,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)
        all_pos_x.append(int(np.round(kp.pt[0])))
        all_pos_y.append(int(np.round(kp.pt[1])))


    all_pos=np.vstack((all_pos_x,all_pos_y)).T
    
    return all_pos, dilated
        
        
#
#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic')
#file_name_of_picture='video4_1211.png'   #'pica33.png'
##
##os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')
##file_name_of_picture='pica36.png'   #'pica33.png'
#
#
#img = cv2.imread(file_name_of_picture,1)
#
#plt.close()
#
#
#img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.figure('original')
#imgplot = plt.imshow(img_RGB)
#blurred = cv2.GaussianBlur(img, (11, 11), 0)
#hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#cv2.imshow('hsv',hsv)
#plt.figure('HSV')
#hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
#imgplot = plt.imshow(hsv_rgb)
#
#
#circles_h=h_method(img,1,150,130,14,5,40)
#circles_s=s_method(img,1,150,130,15,5,40)
#circles_v=v_method(img,1,150,150,15,5,40)
#circles_gray=gray_method(file_name_of_picture,1,150,200,10,10,40)
#
#
#all_pos_x,all_pos_y,all_radius=concatenate_results(method1=circles_h,method2=circles_s,method3=circles_v,method4=circles_gray)
#
#
#all_pos_valid_oldpos_x=Discard_if_too_far_from_old_pos(all_pos=all_pos_x,oldpos=260,thres_oldpos=50)
#all_pos_valid_oldpos_y=Discard_if_too_far_from_old_pos(all_pos=all_pos_y,oldpos=248,thres_oldpos=50)
#all_radius_valid_old=Discard_if_too_far_from_old_pos(all_pos=all_radius,oldpos=12,thres_oldpos=10)
#
#
#
#avg_pos_x=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_valid_oldpos_x,thres_avg_pos=10)
#avg_pos_y=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_valid_oldpos_y,thres_avg_pos=10)
#avg_radius=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_radius_valid_old,thres_avg_pos=5)
#
#
#
#print('avgpos_x',avg_pos_x)
#print('avgpos_y',avg_pos_y)
#print('avg_radius',avg_radius)
#
#"""
#lav et plot på bilelde med alle de muligede position og så avg
#"""
#
#avg_all=np.concatenate((int(avg_pos_x),int(avg_pos_y),int(avg_radius)), axis=None)
##avg_all=np.concatenate((all_pos_x,all_pos_y,all_radius), axis=2)
#avg_all=np.vstack((all_pos_x,all_pos_y,all_radius)).T
#
#avg_all = np.uint16(np.around(avg_all))
#for i in avg_all[:]:
#    # draw the outer circle
#    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
#    # draw the center of the circle
#    cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
#
#cv2.circle(img,(avg_all[0],avg_all[1]),avg_all[2],(255,0,0),12)
## draw the center of the circle
#cv2.circle(img,(avg_all[0],avg_all[1]),2,(255,255,0),8)
#    
#cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.figure('avg_all')
#imgplot = plt.imshow(cimg)

