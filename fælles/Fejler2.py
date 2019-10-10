

import cv2, time
import AlbertFunctions as AF



#Defining input/output folders and image format
input_img_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Data\pic"
output_img_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Data\vid"
start_folder=r"C:\Users\JAlbe\OneDrive\Drone projekt\Scripts\scripts\fælles"
image_format = '.png'

#First, get the files:
files = AF.img_loader(input_img_folder,image_format,sort=True)

#image list
circles = []
img_marked = []
run_times = []

#Variables
img_No = 0
save_images = False

for file in files:    
    start_time = time.time()
    img = cv2.imread((input_img_folder+r"/"+file),cv2.IMREAD_UNCHANGED)
#    if cimg.shape[0] != 480:
#        cimg = cv2.resize(cimg, (640,480))
#    cimg = AF.rotateImage(cimg,180)

    circles = AF.h_circles(img, True ,[])
    h_img = AF.draw_circles(img,circles)
    end_time = time.time()
    run_times.append(end_time-start_time)
    print("Time per frame: " + str(end_time-start_time))
    #Save marked images
    if(save_images):
      AF.img_marked_saver(output_img_folder,image_format,img_No,h_img)
      img_No+= 1    
    img_marked.append(h_img)

#Close windows
#cv2.waitKey(0)
cv2.destroyAllWindows()
print("Avg. time pr frame: "+str(sum(run_times)/len(run_times)))
print("Framerate: "+str(1/(sum(run_times)/len(run_times))))

print(AF.video_export_v2(output_img_folder,img_marked,r"BlaBla.avi"))
