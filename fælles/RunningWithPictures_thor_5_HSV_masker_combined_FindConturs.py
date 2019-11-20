import cv2, os, numpy as np
import AlbertFunctions as AF
import ThorFunctions3 as TH



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
        #red_mask_pp=TH.PreProcessing(red_mask)
        red_mask_pp=TH.PreProcessing(red_mask,number_of_iterations=1)

        red_mask_pp_marked_three_channels = cv2.cvtColor(red_mask_pp,cv2.COLOR_GRAY2RGB) 

        findContour_pos=TH.findContours(red_mask_pp)
        
        findContour_pos = np.uint16(np.around(findContour_pos))
        for i in findContour_pos[0,:]:
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
