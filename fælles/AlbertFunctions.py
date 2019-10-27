#Albert functions.py
import os, numpy as np, glob, re, cv2

#Homemade functions
def img_loader(input_img_folder,image_format,sort):
    try:
        cwd = os.getcwd()
        os.chdir(input_img_folder)
        files =glob.glob1(input_img_folder,'*'+image_format)
        if sort==True:
            files = sorted(files, key=lambda x:float(re.findall(r"(\d+)",x)[0]))
        os.chdir(cwd)
        return files
    except:
        os.chdir(cwd)
        return None

def img_marked_saver(img_folder,image_format,img_No, img):
    try:
        cwd = os.getcwd()    
        os.chdir(img_folder)
        cv2.imwrite(str(img_No)+image_format, img)
        os.chdir(cwd)
        return True
    except:
        os.chdir(cwd)
        print('Could not save image')
        return False

def ball_drift(center, previouscenter, threshold):
    try: 
        ball_drift = np.sqrt(np.sum((center-previouscenter)**2))
        if ball_drift < threshold:
            print('Ball_drift = ',ball_drift)
            return True
        else:
            return False
    except:
        return None

def ball_size(ball_size, pixel_size):
    try:
        #Z = f*H/h, Z = distance to object, H = ball size, h = pixel size , f = focal length 3,04 mm 
        return pixel_size/ball_size
    except:
        return False

def video_export_v1(output_img_folder,image_format,sort):
    img_array = []
    try:
        os.chdir(output_img_folder)
        files =glob.glob1(output_img_folder,'*'+image_format)
        if sort==True:
            files = sorted(files, key=lambda x:float(re.findall(r"(\d+)",x)[0]))
        
        for file in files:
            img = cv2.imread(file)
            height, width, = img.shape
            size = (width,height)
            img_array.append(img)
        print(len(img_array))
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()
        return 1
    except:
        return None


#Parameters:	
#image – 8-bit, single-channel, grayscale input image.
#circles – Output vector of found circles. Each vector is encoded as a 3-element floating-p                                                                                    oint vector (x, y, radius) .
#circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
#method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
#dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.

#for filename in glob.glob("*.png"): # This line take all the files of the filename .png from the current folder. Source http://stackoverflow.com/questions/6997419/how-to-create-a-loop-to-read-several-images-in-a-python-script
# if you want sort files according to the digits included in the filename, you can do as following:
#files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))

#HoughCircles params:
#minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
#param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
#param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
#minRadius – Minimum circle radius.
#maxRadius – Maximum circle radius

#For basement pictures in TestDrone640 use minRadius=5,maxRadius=30, param1 = 500, param2 = 9,minDist=500)
#For  pica 30 use: param1=500,param2=9,minRadius=5,maxRadius=15) 
def h_circles(img, blur,params):
    try:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if blur:
            img = cv2.GaussianBlur(img,(3,3),0)
        if params == []:
            params=[500, 200, 5, 15, 80]        
#            print("No params provided")
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=params[0],param1=params[1],param2=params[2],minRadius=params[3],maxRadius=params[4])
        circles = circles.astype(int)
        return circles
    except:
        return None

def draw_circles(img, circ):
    try:
      circ = np.uint16(np.around(circ))
      for i in circ[0,:]:
              # draw the outer circle
              cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
              # draw the center of the circle
              cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
      return img
    except:
        print('No circles found in image: ')
        return img

def draw_circles2(img, circ):
    try:
      circ = np.uint16(np.around(circ))
      # draw the outer circle
      cv2.circle(img,(circ[0],circ[1]),circ[2],(0,255,0),2)
      # draw the center of the circle
      cv2.circle(img,(circ[0],circ[1]),2,(0,0,255),3)
      return img
    except:
        print('No circles found in image: ')
        return img


def video_export_v2(output_img_folder,images,filename):
    
    cwd = os.getcwd()
    os.chdir(output_img_folder)
    img_array = []
#    try:
    for img in images:
        height, width, layer = img.shape 
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(cwd)
    os.chdir(cwd)
    return True
#    except:
#        os.chdir(cwd)           
#        return None

def colourmask(img,colour):
#Different applications use different scales for HSV. 
#For example gimp uses H = 0-360, S = 0-100 and V = 0-100. 
#But OpenCV uses H: 0-179, S: 0-255, V: 0-255. Here i got a hue value of 22 in gimp.
#So I took half of it, 11, and defined range for that. ie (5,50,50) - (15,255,255).
#    masked_img=[]
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if colour == 'red':
            low_red1 = np.array([0, 200, 50])
#            high_red1 = np.array([10, 255, 255])
            high_red1 = np.array([20, 255, 255])
            low_red2 = np.array([175, 200, 50])
            high_red2 = np.array([179, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, low_red1, high_red1)
#            cv2.imshow("red_lower",red_mask1)
            red_mask2 = cv2.inRange(img_hsv, low_red2, high_red2)
#            cv2.imshow("red_higher",red_mask2)
#            masked_img1 = cv2.bitwise_and(img, img, mask=red_mask1)
#            masked_img2 = cv2.bitwise_and(img, img, mask=red_mask2)
#            masked_img = cv2.bitwise_or(masked_img1, masked_img2)            
#            return masked_img
            return cv2.bitwise_or(red_mask1,red_mask2)
        elif colour == 'green':
            low_green = np.array([38, 50, 50])
            high_green = np.array([75, 255, 255])
            green_mask = cv2.inRange(img_hsv, low_green, high_green)
#            masked_img = cv2.bitwise_and(img, img, mask=green_mask)
#            return masked_img
            return green_mask
        elif colour == 'blue':
            low_blue = np.array([94, 80, 2])
            high_blue = np.array([126, 255, 255])
            blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)
#            masked_img = cv2.bitwise_and(img, img, mask=blue_mask)
#            return masked_img
            return blue_mask
        
    except:
        return None

def search_box1(img,yxr,scale):
#    print("img: ",img, "yxr: ",yxr)
    size=np.uint16(np.around(scale*yxr[2]))
    xL = yxr[1]-size
    xH = yxr[1]+size
    yL = yxr[0]-size
    yH = yxr[0]+size
    if(xH>img[0]):
        xH=img[0]
    if(xL<0):
        xL=0
    if(yH>img[1]):
        yH=img[1]
    if(yL<0):
        yL=0
#    print("xL: ",xL,"xH: ",xH,"yL: ",yL,"yH: ",yH)
    return [xL,xH,yL,yH]

def search_box2(img,yxr,scale):
#    print("img: ",img.shape, "yxr: ",yxr)
    size=np.uint16(np.around(scale*yxr[2]))
    xL = yxr[1]-size
    xH = yxr[1]+size
    yL = yxr[0]-size
    yH = yxr[0]+size
    if(xH>img.shape[0]):
        xH=img.shape[0]
    if(xL<0):
        xL=0
    if(yH>img.shape[1]):
        yH=img.shape[1]
    if(yL<0):
        yL=0
    b_img=cv2.rectangle(img, (yL, xL), (yH, xH), (255,0,0), 2)
#    print("xL: ",xL,"xH: ",xH,"yL: ",yL,"yH: ",yH)
    return b_img

def search_colour(mask_img,box):
    value = sum(sum(mask_img[box[0]:box[1],box[2]:box[3]]))
    return value


# ========================================================================#
#TBD
def ball_dist(radius,ballsize):
  return(radius)


#https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result