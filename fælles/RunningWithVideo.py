import cv2
import numpy as np
import time
import AlbertFunctions as AF
import imutils
#name = sys.argv[1]

#name = 'ball_track.mp4'
path_to_video = r"C:\Users\JAlbe\OneDrive\Drone projekt\Data"
name = "flight4_red.mp4"
#name = '358_ball_lost.mp4'
minDist = 500 # 500
param1 = 200 #500
param2 = 5 #was  #12
minRadius = 15 # was 5
maxRadius = 80 # was 50


cap = cv2.VideoCapture((path_to_video+r"/"+name))
  

# Raspicam
#camMatrix = np.array( [[633.06058204 ,  0.0  ,       330.28981083], [  0.0,  631.01252673 ,226.42308878], [  0.0, 0.0,1.        ]])
#distCoefs = np.array([ 5.03468649e-02 ,-4.38421987e-02 ,-2.52895273e-04 , 1.91361583e-03, -4.90955908e-01])

#ananda phone
#camMatrix = np.array( [[630.029356,   0 , 317.89685204], [  0.  ,  631.62683668 ,242.01760626], [  0.  ,  0.,   1.  ]] )
#distCoefs =  np.array([ 0.318628685 ,-2.22790350 ,-0.00156275882 ,-0.00149764901,  4.84589387])

circles = []
frames = 0

while cap.isOpened() :

    start_time = time.time()

    ret, cimg = cap.read()
    cimg = AF.rotateImage(cimg,180)
    
    if not ret:
        break
      
    if cimg.shape[0] != 480:
#     cimg = cv2.resize(cimg, (640,480))
      cimg = imutils.resize(cimg,width = cimg.shape[0],height = cimg.shape[1])
    
    img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
    img2= cv2.Canny(img,100,200)
    img2 = cv2.GaussianBlur(img2,(3,3),0)
    img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.erode(img, None, iterations=1)
    #img = cv2.dilate(img, None, iterations=1)

    #Find circles
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)

    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),2)
            cv2.circle(img2,(i[0],i[1]),i[2],(255,255,255),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            cv2.circle(img,(i[0],i[1]),2,(255,255,255),3)
            cv2.circle(img2,(i[0],i[1]),2,(255,255,255),3)
    except:
        print('No circles found in image: ')
        flag_BallFound = False 

    
    
    frames += 1
    
    
    
    #Show er ikke talt med i computational tid, da de ikke skal bruges når
    #det køres på dronen
    end_time = time.time()
    print("Time per frame: " + str(end_time-start_time))
    
    #Freq virker ikke, da der ikke er noget computational tid i øjeblikket,
    #Så den prøver at dividere med 0
    #print("Freq = " + str( 1/(time.time() - start_time) ))
    
    #show_1 = np.hstack( ( orig_im, track_im ) )
    #show_2 = np.hstack( ( gray_show, hsv_range_show ) )
    
    #Hvis i vil vise flere forskellige typer frames på samme tid. Kan også
    #gøres med flere imshows, men her hænger billedet sammen og skygger
    #ikke for hinanden
    img_show=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    show_1 = np.hstack( ( cimg, cimg ) )
    show_2 = np.hstack( ( img_show, img2_show ) )

    show_f = np.vstack( (show_1, show_2) )
    
    #0,0,0 Indsæt koordinaterne for bolden
    text = "( {:.2f} | {:.2f} | {:.2f}  )".format(0,0,0)
    cv2.putText(show_f, text, (640,480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    

    cv2.imshow('Result', show_f)

    if( cv2.waitKey( 1 ) & 0xFF == ord('q') ):
        break;

print("Number of frames in the video: " + str(frames))
        
cap.release()    
cv2.destroyAllWindows()
