import cv2, os, numpy as np
import AlbertFunctions as AF
import ThorFunctions as TH



#Defining input/output folders and image format
input_img_folder=r"C:\Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_as_pic"
output_img_folder=r"C:\Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video1_output"
start_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Scripts\scripts\fælles"
image_format = '.png'
#First, get the files:
files = AF.img_loader(input_img_folder,image_format,True)



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


save_images = True

counter=0

for file in files:    
    counter+=1
    if counter>500:
        break    
    print(file)

    try:
        img = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
      
        
        red_mask=TH.colourmask(img,'red')
        
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
        eroded = cv2.erode(red_mask, element, iterations=3)
        dilated = cv2.dilate(eroded, element, iterations=10)
        
        # blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.blobColor = 0
        #params.minThreshold = 14
        #params.maxThreshold = 25
        params.minDistBetweenBlobs=1000
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
        i = 0
        for kp in keypts:
    #        print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
            i+=1
    #        cv2.rectangle(res,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)
            all_pos_x.append(int(np.round(kp.pt[0])))
            all_pos_y.append(int(np.round(kp.pt[1])))
    
    
        all_pos=np.vstack((all_pos_x,all_pos_y)).T
        
    
    
        all_pos = np.uint16(np.around(all_pos))
        for i in all_pos[:]:
            # draw the outer circle
    #            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,255,0),10)
        
    #        cv2.circle(img,(avg_all[0],avg_all[1]),avg_all[2],(255,0,0),12)
        # draw the center of the circle
    #        cv2.circle(img,(avg_all[0],avg_all[1]),2,(255,255,0),8)
    #            consec_balls_found = consec_balls_found + 1
    #            if consec_balls_found > minConsecBallsFound:
    #                flag_BallFound = True
    except: 
        print('No circles found in image: ',file)
#        flag_BallFound = False 
#        consec_balls_found = 0
    #Save marked images
    if(save_images):
      try:
          AF.img_marked_saver(output_img_folder,image_format,img_No,img)
      except:
          print('Could not save image')
    img_No= img_No + 1
    img_marked.append(img)
    


#Close windows
#cv2.waitKey(0)
cv2.destroyAllWindows()

#Make a video
#AF.video_export_v1(output_img_folder,image_format,True)

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output')
print(AF.video_export_v2(output_img_folder,img_marked,r"BlaBla.avi"))
