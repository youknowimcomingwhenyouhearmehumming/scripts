import cv2
import os

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/')



cv2.namedWindow('w')
cam0=cv2.VideoCapture('video4.mov')
i=0
j=0

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic')


while True:
  (ret,img)=cam0.read()
  img = cv2.resize(img, (640,480))
  cv2.imshow('w',img)
  k=cv2.waitKey(1)
#  print(k)
  j+=1
  if j%1==0:
     i=i+1
     cv2.imwrite('video_'+str(i)+'.png',img);
#  if k%256==27:
#    break


#stopped with 1414