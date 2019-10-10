import cv2, os, numpy as np
import AlbertFunctions as AF
import ThorFunctions as TH



#Defining input/output folders and image format
input_img_folder=r"C:\Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_as_pic"
output_img_folder=r"C:\Users/Bruger/Documents/Uni/Abu dhabi/data/newvideo/video4_output"
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
    if counter>10:
        break     

    img = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
#    if cimg.shape[0] != 480:
#        cimg = cv2.resize(cimg, (640,480))
#    cimg = AF.rotateImage(cimg,180)
#    img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
#    img = cv2.GaussianBlur(img,(11,11),0)
    #img = cv2.erode(img, None, iterations=1)
    #img = cv2.dilate(img, None, iterations=1)


    circles_h=TH.h_method(img,1,150,130,14,5,40)
    circles_s=TH.s_method(img,1,150,130,15,5,40)
    circles_v=TH.v_method(img,1,150,150,15,5,40)
    circles_gray=TH.gray_method((input_img_folder+r"/"+file),1,150,200,10,10,40)
    
    all_pos_x,all_pos_y,all_radius=TH.concatenate_results(method1=circles_h,method2=circles_s,method3=circles_v,method4=circles_gray)
    
#    all_pos_valid_oldpos_x=Discard_if_too_far_from_old_pos(all_pos=all_pos_x,oldpos=260,thres_oldpos=50)
#    all_pos_valid_oldpos_y=Discard_if_too_far_from_old_pos(all_pos=all_pos_y,oldpos=248,thres_oldpos=50)
#    all_radius_valid_old=Discard_if_too_far_from_old_pos(all_pos=all_radius,oldpos=12,thres_oldpos=10)
        
    
    avg_pos_x_final=TH.Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_x,thres_avg_pos=100)
    avg_pos_y_final=TH.Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_pos_y,thres_avg_pos=100)
    avg_radius_final=TH.Discard_outlier_and_find_mean_pos(all_pos_valid_oldpos=all_radius,thres_avg_pos=5)

    print('avgpos_x',avg_pos_x_final)
    print('avgpos_y',avg_pos_y_final)
    print('avg_radius_final',avg_radius_final)
    
    
    """
    lav et plot på bilelde med alle de muligede position og så avg_final
    """
    
    avg_all_final=np.concatenate((int(avg_pos_x_final),int(avg_pos_y_final),int(avg_radius_final)), axis=None)
    #avg_all=np.concatenate((all_pos_x,all_pos_y,all_radius), axis=2)
    avg_all=np.vstack((all_pos_x,all_pos_y,all_radius)).T
    

    
    try:
        avg_all = np.uint16(np.around(avg_all))
        for i in avg_all[:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),12)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),8)
        
        cv2.circle(img,(avg_all_final[0],avg_all_final[1]),avg_all_final[2],(255,0,0),12)
        # draw the center of the circle
        cv2.circle(img,(avg_all_final[0],avg_all_final[1]),2,(255,255,0),8)
#            consec_balls_found = consec_balls_found + 1
#            if consec_balls_found > minConsecBallsFound:
#                flag_BallFound = True
    except:
        print('No circles found in image: ',file)
        flag_BallFound = False 
        consec_balls_found = 0
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
