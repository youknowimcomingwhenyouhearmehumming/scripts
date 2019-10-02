import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob 
import re


#Homemade functions
def img_loader(input_img_folder,image_format,sort):
    try:
        os.chdir(input_img_folder)
        files =glob.glob1(input_img_folder,'*'+image_format)
        if sort==True:
            files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
        return files
    except:
        return None

def img_marked_saver(img_folder,image_format,img_No, img):
    try:    
        os.chdir(output_img_folder)
        cv2.imwrite(str(img_No)+image_format, img)
        return True
    except:
        return False

def ball_drift(center, previouscenter, threshold):
    try: 
        ball_drift = center-previouscenter
        if ball_drift < threshold:
            return True
        else:
            return flag
    except:
        return None

def ball_size(ball_size, pixel_size):
    try:
        #Z = f*H/h, Z = distance to object, H = ball size, h = pixel size , f = focal length 3,04 mm 
        return pixel_size/ball_size
    except:
        return False

def video_export_v1(output_img_folder,image_format,sort):
    img_array = []
    try:
        os.chdir(output_img_folder)
        files =glob.glob1(output_img_folder,'*'+image_format)
        if sort==True:
            files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
        
        for file in files:
            img = cv2.imread(file)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        print(len(img_array))
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()
        return 1
    except:
        return None

def video_export_v2(output_img_folder,images):
    img_array = []
    try:
        for img in images:
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter('project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        return 1
    except:
        return None

#https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result




#Defining input/output folders and image format
input_img_folder=r'C:/Users/JAlbe/OneDrive/Drone projekt/Albert/Basement_Pictures/Test1/Raw'
output_img_folder=r'C:/Users/JAlbe/OneDrive/Drone projekt/Albert/Basement_Pictures/Test1/Marked'
image_format = '.png'

#First, get the files:
files = img_loader(input_img_folder,image_format,True)

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

#For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
 
minDist = 500 # 500
param1 = 500 #500
param2 = 12 #was  #12
minRadius = 5
maxRadius = 30 # was 30

#Confidence needed
minConsecBallsFound = 5
consec_balls_found = 0
flag_BallFound = False

#image list
circles = []
img_marked = []


#Variables
img_No=0

for file in files:    
    os.chdir(input_img_folder)
#    print('Current frame: ',file)
    #                                                                                                               #
    cimg = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    cimg = rotateImage(cimg,180)
    img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(img,(11,11),0)
    #img = cv2.erode(img, None, iterations=1)
    #img = cv2.dilate(img, None, iterations=1)


    #                                                                                                               #
    #Find circles
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)

    try:
      circles = np.uint16(np.around(circles))
      for i in circles[0,:]:
          # draw the outer circle
          cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
          # draw the center of the circle
          cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#          print(consec_balls_found)
          consec_balls_found = consec_balls_found + 1
          if consec_balls_found > minConsecBallsFound:
              flag_BallFound = True
    except:
#        print('No circles found in image: ',file)
        flag_BallFound = False 
        consec_balls_found = 0
    
    
    #Save marked images
    try:
        img_marked_saver(output_img_folder,image_format,img_No,cimg)
    except:
        print('Could not save image')
    img_No= img_No + 1
    img_marked.append(cimg)
    
    
    
#Close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imshow('tester',img_marked[0][1,:])
#Make a video
video_export_v1(output_img_folder,image_format,True)
video_export_v2(output_img_folder,img_marked)

#for i in range(10):
#    print(img_marked[:][i])
#    
#print(img_marked[0][0][0][0])
#print(cimg)

