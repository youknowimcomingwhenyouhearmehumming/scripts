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
    if counter>200:
        break    
    print(file)

    try:
        img = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
  
    
        img_red_mask=TH.colourmask(img,'red')
        
    except:
        print('No circles found in image: ',file)
        flag_BallFound = False 
        consec_balls_found = 0
    #Save marked images
    if(save_images):
      try:
          AF.img_marked_saver(output_img_folder,image_format,img_No,img_red_mask)
      except:
          print('Could not save image')
    img_No= img_No + 1
    img_marked.append(img_red_mask)
    


#Close windows
#cv2.waitKey(0)
cv2.destroyAllWindows()

#Make a video
#AF.video_export_v1(output_img_folder,image_format,True)

#os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output')
print(AF.video_export_v2(output_img_folder,img_marked,r"BlaBla.avi"))
