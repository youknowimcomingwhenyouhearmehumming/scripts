import cv2
import imutils
import os

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/ballpictures_jesper/TestDrone640_2')

img = cv2.imread('image_98.png')

img2 = cv2.imread('image_62.png')


#
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
##img=mpimg.imread('image_53.png')
#imgplot = plt.imshow(img)
#
#cv::Mat diffImage;
#cv::absdiff(img, im2, diffImage);