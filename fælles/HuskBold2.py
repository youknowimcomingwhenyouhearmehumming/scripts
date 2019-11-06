# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:33:38 2019

@author: JAlbe
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:30:19 2019

@author: JAlbe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:33:31 2019

@author: JAlbe
"""

import cv2
import numpy as np
import time
import AlbertFunctions as AF
import imutils



path_to_video = r"C:\Users\JAlbe\OneDrive\Drone projekt\Data"
path_to_video = r"C:\Users\JAlbe\Documents\GitHub\Data"
name = "flight4_red.mp4"

cap = cv2.VideoCapture((path_to_video+r"/"+name))


size = 600
CandCirc = []
CandBox=np.zeros((3,4))
xLxHyLyH = np.zeros((1,4))

left = 0
right = 0
up = 0
down = 0

counterFrames = 0
counterRegion = 0

isBall = False

while cap.isOpened() :
    timeStart = time.time()
    ret, imgOrg = cap.read()
    try:
        imgOrg = AF.rotateImage(imgOrg,180)
    except:
        break
    if not ret:
        break      
    if imgOrg.shape[0] != 480:
#     cimg = cv2.resize(cimg, (640,480))
      imgOrg = imutils.resize(imgOrg,width = imgOrg.shape[0],height = imgOrg.shape[1])

    imgMask = AF.colourmask(imgOrg,"red")
    imgMask = cv2.erode(imgMask, None, iterations=1)
    imgMask = cv2.dilate(imgMask, None, iterations=1)
    
    ############################################################# End of making images
    
    if(isBall):
        CandCirc = AF.h_circles(imgOrg[xLxHyLyH[0]:xLxHyLyH[1],xLxHyLyH[2]:xLxHyLyH[3]], True ,[])
        cv2.imshow('Cropped', imgOrg[xLxHyLyH[0]:xLxHyLyH[1],xLxHyLyH[2]:xLxHyLyH[3]])
        try:
            for circ in range(CandCirc.shape[1]):
    #            value = sum(sum(imgMask[CandBox[circ][0]:CandBox[circ][1],CandBox[circ][2]:CandBox[circ][3]]))
                imgOrg = AF.draw_circles(imgOrg,CandCirc[0][circ][:])       
        except: 
            counterRegion += 1 
            isBall = False
    else:
        CandCirc = AF.h_circles(imgOrg, True ,[])
        for circ in range(CandCirc.shape[1]):
            [left,right,up,down] = AF.getBoxLim(imgOrg.shape,CandCirc[0][circ][:],1.2)
            left=int(left)
            right=int(right)
            up=int(up)
            down=int(down)
            value = sum(sum(imgMask[up:down][left:right]))
            if(value>255*20):
                isBall = True
                imgOrg = AF.draw_circles2(imgOrg,CandCirc[0][circ][:])
                xLxHyLyH[:] = CandBox[circ][:]
                xLxHyLyH = xLxHyLyH.astype(int)
                print("xLxHyLyH:")
                print(xLxHyLyH)
                imgOrg = cv2.rectangle(imgOrg,(xLxHyLyH[2],xLxHyLyH[0]), (xLxHyLyH[3],xLxHyLyH[1]),(255,0,0),2)
                break
            else:
                isBall = False

    timeEnd = time.time()
    counterFrames += 1
    cv2.imshow('full', imgOrg)
    cv2.imshow('mask', imgMask)
#    print("Time per frame: " + str(round(timeEnd-timeStart,3)) + "\t isBall: " + str(isBall))
    
    
    if( cv2.waitKey( 1 ) & 0xFF == ord('q') ):
        break;
        
print("Number of counterFrames in the video: " + str(counterFrames))
        
cap.release()    
cv2.destroyAllWindows()
