

import cv2, os, numpy as np
import AlbertFunctions as AF
import ThorFunctions as TH
import glob 
from PIL import Image

#Defining input/output folders and image format
input_img_folder=r'C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic'
output_img_folder=r'C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output'
image_format = '.png'

#First, get the files:
files = AF.img_loader(input_img_folder,image_format,True)

#Parameters:	
#image – 8-bit, single-channel, grayscale input image.
#circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x, y, radius) .
#circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
#method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
#dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.

#for filename in glob.glob("*.png"): # This line take all the files of the filename .png from the current folder. Source http://stackoverflow.com/questions/6997419/how-to-create-a-loop-to-read-several-images-in-a-python-script
# if you want sort files according to the digits included in the filename, you can do as following:
#files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))

#HoughCircles params:
#minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
#param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
#param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
#minRadius – Minimum circle radius.
#maxRadius – Maximum circle radius

#For basement pictures in TestDrone640 use minRadius=5,maxRadius=30, param1 = 500, param2 = 9,minDist=500)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
 
minDist = 500 # 500
param1 = 200 #500
param2 = 5 #was  #12
minRadius = 15 # was 5
maxRadius = 50 # was 30

#Confidence needed
minConsecBallsFound = 5
consec_balls_found = 0
flag_BallFound = False

#image list
circles = []
img_marked = []
    

#Variables
img_No = 0
counter = 0

counter=0

os.chdir(input_img_folder)


for file in glob.glob("*.png"): # This line take all the files of the filename .png from the current folder. Source http://stackoverflow.com/questions/6997419/how-to-create-a-loop-to-read-several-images-in-a-python-script
#    col=Image.open(filename)

#for file in files:    
    counter+=1
    if counter>90:
        break 
    print(file)
#    cimg = cv2.imread(file,1)
#
##img_RGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##        
##    cimg = cv2.imread((file),cv2.IMREAD_UNCHANGED)
##    print(file)
#    img= cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    try:        
    #plt.figure('original')
    #imgplot = plt.imshow(img_RGB)
        img_blurred = cv2.GaussianBlur(img, (11, 11), 0)
    #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)
    #plt.figure('HSV')
    #hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    #imgplot = plt.imshow(hsv_rgb)
    
    
        circles_h=TH.h_method(img_blurred,1,150,130,14,5,40)
        circles_s=s_method(img_blurred,1,150,130,15,5,40)
        circles_v=v_method(img_blurred,1,150,150,15,5,40)
        circles_gray=gray_method(str(file),1,150,200,10,10,40)
    
    
        all_pos_x,all_pos_y,all_radius=concatenate_results(method1=circles_h,method2=circles_s,method3=circles_v,method4=circles_gray)
    
        
    #    all_pos_valid_oldpos_x=Discard_if_too_far_from_old_pos(all_pos=all_pos_x,oldpos=260,thres_oldpos=50)
    #    all_pos_valid_oldpos_y=Discard_if_too_far_from_old_pos(all_pos=all_pos_y,oldpos=248,thres_oldpos=50)
    #    all_radius_valid_old=Discard_if_too_far_from_old_pos(all_pos=all_radius,oldpos=12,thres_oldpos=10)
    #    
    
    
        avg_pos_x_final=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_x,thres_avg_pos=30)
        avg_pos_y_final=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_y,thres_avg_pos=30)
        avg_radius_final=Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_radius,thres_avg_pos=10)
        
    
    
        print('avgpos_x',avg_pos_x_final)
        print('avgpos_y',avg_pos_y_final)
        print('avg_radius_final',avg_radius_final)
    
    #"""
    #lav et plot på bilelde med alle de muligede position og så avg_final
    #"""
    
        avg_all_final=np.concatenate((int(avg_pos_x_final),int(avg_pos_y_final),int(avg_radius_final)), axis=None)
        #avg_all=np.concatenate((all_pos_x,all_pos_y,all_radius), axis=2)
        avg_all=np.vstack((all_pos_x,all_pos_y,all_radius)).T
    
        avg_all = np.uint16(np.around(avg_all))
        for i in avg_all[:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
        
        cv2.circle(img,(avg_all_final[0],avg_all_final[1]),avg_all_final[2],(255,0,0),12)
        # draw the center of the circle
        cv2.circle(img,(avg_all_final[0],avg_all_final[1]),2,(255,255,0),8)
        
        cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #plt.figure('avg_all')
    #imgplot = plt.imshow(cimg)
    

    #    try:
    #        circles = np.uint16(np.around(circles))
    #        for i in circles[0,:]:
    #            # draw the outer circle
    #            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #            # draw the center of the circle
    #            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #            consec_balls_found = consec_balls_found + 1
    #            if consec_balls_found > minConsecBallsFound:
    #                flag_BallFound = True
    except:
        print('No circles found in image: ',file)
        flag_BallFound = False 
        consec_balls_found = 0
    #Save marked images
    try:
        AF.img_marked_saver(output_img_folder,image_format,img_No,cimg)
    except:
        print('Could not save image')
    
    img_No= img_No + 1
    try:
        img_marked.append(cimg)
    except:
#        img_marked.append(img)
        pass
#    cv2.imshow('output',cimg)
#    cv2.waitKey(0)
    
#cv2.imshow('output',cimg)
##Close windows
##cv2.waitKey(0)
#cv2.destroyAllWindows()

#Make a video
#AF.video_export_v1(output_img_folder,image_format,True)

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output')
AF.video_export_v2(output_img_folder,img_marked)
