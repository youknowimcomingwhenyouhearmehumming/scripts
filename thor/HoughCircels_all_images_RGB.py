import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob 
from PIL import Image

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/outdoor')


for filename in sorted(glob.glob('*.png'), key=os.path.getmtime):
    
#for filename in glob.glob("*.png"): # This line take all the files of the filename .png from the current folder. Source http://stackoverflow.com/questions/6997419/how-to-create-a-loop-to-read-several-images-in-a-python-script

    col=Image.open(filename)
    img = cv2.imread(filename,0)
    print (filename)
#    img = cv2.imread('pica1.png',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #cimg = cv2.erode(cimg, None, iterations=1)
    #cimg = cv2.dilate(cimg, None, iterations=1)
    
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=1000,
    param1=500,param2=9,minRadius=10,maxRadius=40) 
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=50,
    param1=500,param2=12,minRadius=5,maxRadius=40) 
    #For basement pictures in TestDrone640 use minRadius=10,maxRadius=30)
    #For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
    
    try: 
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        #cv2.imshow('detected circles',cimg)
    except:
        print('File',filename,'did not found any circel in it')
    cv2.imshow('r',cimg)
    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    imgplot = plt.imshow(cimg)
    #
    #cv2.destroyAllWindows()
    
