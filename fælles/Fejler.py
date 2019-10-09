

import cv2, os, numpy as np
import AlbertFunctions as AF



#Defining input/output folders and image format
input_img_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Data\pic"
output_img_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Data\vid"
start_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Scripts\scripts\fælles"
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


save_images = False

for file in files:    
    cimg = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
#    if cimg.shape[0] != 480:
#        cimg = cv2.resize(cimg, (640,480))
#    cimg = AF.rotateImage(cimg,180)
    img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(11,11),0)
    #img = cv2.erode(img, None, iterations=1)
    #img = cv2.dilate(img, None, iterations=1)

    #Find circles
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            consec_balls_found = consec_balls_found + 1
            if consec_balls_found > minConsecBallsFound:
                flag_BallFound = True
    except:
        print('No circles found in image: ',file)
        flag_BallFound = False 
        consec_balls_found = 0
    #Save marked images
    if(save_images):
      try:
          AF.img_marked_saver(output_img_folder,image_format,img_No,cimg)
      except:
          print('Could not save image')
    img_No= img_No + 1
    img_marked.append(cimg)



#Close windows
#cv2.waitKey(0)
cv2.destroyAllWindows()

#Make a video
#AF.video_export_v1(output_img_folder,image_format,True)

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output')
print(AF.video_export_v2(output_img_folder,img_marked,r"BlaBla.avi"))
