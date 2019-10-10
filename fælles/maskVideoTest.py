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

    ret, org_img = cap.read()
    org_img = AF.rotateImage(org_img,180)
    
    if not ret:
        break
      
    if org_img.shape[0] != 480:
#     cimg = cv2.resize(cimg, (640,480))
      org_img = imutils.resize(org_img,width = org_img.shape[0],height = org_img.shape[1])
    mask_img = AF.colourmask(org_img,"red")
    circles = AF.h_circles(org_img, True ,[])
    h_img = AF.draw_circles(org_img,circles)
    
    



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
    
    img2_show = cv2.cvtColor(h_img,cv2.COLOR_BGR2GRAY)
    
    show_1 = np.hstack( ( org_img, h_img ) )
    show_2 = np.hstack( ( mask_img, img2_show ) )

#    show_f = np.vstack( (show_1, show_2) )
    
    #0,0,0 Indsæt koordinaterne for bolden
#    text = "( {:.2f} | {:.2f} | {:.2f}  )".format(0,0,0)
#    cv2.putText(show_f, text, (640,480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    

#    cv2.imshow('Result', show_f)
    
    cv2.imshow('1', show_1)
    cv2.imshow('2', show_2)

    if( cv2.waitKey( 1 ) & 0xFF == ord('q') ):
        break;

print("Number of frames in the video: " + str(frames))
        
cap.release()    
cv2.destroyAllWindows()
