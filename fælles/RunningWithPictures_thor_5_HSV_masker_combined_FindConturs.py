import cv2, os, numpy as np
import AlbertFunctions as AF
import ThorFunctions2 as TH



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
    if counter>1500:
        break    
    print(file)

    try:
        img = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
      
        
        red_mask=TH.colourmask(img,'red')
        
        red_mask_pp=TH.PreProcessing(red_mask)


        red_mask_pp_marked_three_channels = cv2.cvtColor(red_mask_pp,cv2.COLOR_GRAY2RGB) 
        
    
        
#        Hough_pos = cv2.HoughCircles(red_mask_pp,cv2.HOUGH_GRADIENT,dp=1,minDist=100,param1=400,param2=13,minRadius=3,maxRadius=50) 



        #im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret, thresh = cv2.threshold(red_mask_pp, 1, 255, 0)
        #contours,hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #New lines added from pyimage
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        
        
        	
        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        diameter1=TH.calculateDistance(extLeft,extRight)
        diameter2=TH.calculateDistance(extTop,extBot)
        
        if diameter1 >= diameter2:
            radius=np.round(diameter1/2)
        else:
            radius=np.round(diameter2/2)
        
        
        #Only if you want to draw the extrema
        cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
        cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(img, extRight, 8, (0, 255, 0), -1)
        cv2.circle(img, extTop, 8, (255, 0, 0), -1)
        cv2.circle(img, extBot, 8, (255, 255, 0), -1)
         
        
        
        contours_area = []
        # calculate area and filter into new array
        for con in contours:
            area = cv2.contourArea(con)
            print(area)
            if 100 < area < 40000:
                contours_area.append(con)
                
                
        contours_cirles = []
        
        # check if contour is of circular shape
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            print (circularity)
            if 0.1 < circularity < 1.2:
                contours_cirles.append(con)
                
                
        cv2.drawContours(img, contours_cirles, -1, (0, 255, 0), 3) 
          








        Hough_pos = np.uint16(np.around(Hough_pos))
        

        for i in Hough_pos[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
            cv2.circle(red_mask_pp_marked_three_channels,(i[0],i[1]),i[2],(0,255,0),12)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
            cv2.circle(red_mask_pp_marked_three_channels,(i[0],i[1]),2,(0,0,255),8)

                
                
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
#    img_marked.append(img)
    
    red_mask_three_channels = cv2.cvtColor(red_mask,cv2.COLOR_GRAY2RGB) 
    red_mask_marked_three_channels = cv2.cvtColor(red_mask_pp,cv2.COLOR_GRAY2RGB) 
#    print(np.shape(img),np.shape(red_mask_three_channels))
    show_1 = np.hstack( ( img, red_mask_three_channels ) )
    show_2 = np.hstack( ( red_mask_pp_marked_three_channels, red_mask_marked_three_channels ) )
    show_all = np.vstack( (show_1, show_2) )
    
    cv2.putText(show_all, file, (200,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    img_marked.append(show_all)




#Close windows
#cv2.waitKey(0)
cv2.destroyAllWindows()

#Make a video
#AF.video_export_v1(output_img_folder,image_format,True)

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output')
print(AF.video_export_v2(output_img_folder,img_marked,r"BlaBla.avi"))
