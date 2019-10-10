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

def h_circles(img, blur,params):
    try:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if blur:
            img = cv2.GaussianBlur(img,(3,3),0)
        if params == []:
            params=[500, 200, 5, 15, 80]        
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=params[0],param1=params[1],param2=params[2],minRadius=params[3],maxRadius=params[4])
        return circles
    except:
        return None

def draw_circles(img, circles):
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
                # draw the outer circle
                cv2.circles(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circles(cimg,(i[0],i[1]),2,(0,0,255),3)
        return img
    except:
        print('No circles found in image: ')
        return None



def video_export_v2(output_img_folder,images,filename):
    
    cwd = os.getcwd()
    os.chdir(output_img_folder)
    img_array = []
    try:
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
    except:
        os.chdir(cwd)           
        return None

def colourmask(img,colour):
#Different applications use different scales for HSV. 
#For example gimp uses H = 0-360, S = 0-100 and V = 0-100. 
#But OpenCV uses H: 0-179, S: 0-255, V: 0-255. Here i got a hue value of 22 in gimp.
#So I took half of it, 11, and defined range for that. ie (5,50,50) - (15,255,255).
    masked_img=[]
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if colour == 'red':
            low_red1 = np.array([0, 200, 50])
            high_red1 = np.array([10, 255, 255])
            low_red2 = np.array([175, 200, 50])
            high_red2 = np.array([179, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, low_red1, high_red1)
            red_mask2 = cv2.inRange(img_hsv, low_red2, high_red2)
            masked_img1 = cv2.bitwise_and(img, img, mask=red_mask1)
            masked_img2 = cv2.bitwise_and(img, img, mask=red_mask2)
            masked_img = cv2.bitwise_or(masked_img1, masked_img2)
            
        elif colour == 'green':
            low_green = np.array([38, 50, 50])
            high_green = np.array([75, 255, 255])
            green_mask = cv2.inRange(img_hsv, low_green, high_green)
            masked_img = cv2.bitwise_and(img, img, mask=green_mask)
        elif colour == 'blue':
            low_blue = np.array([94, 80, 2])
            high_blue = np.array([126, 255, 255])
            blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)
            masked_img = cv2.bitwise_and(img, img, mask=blue_mask)
        return masked_img
    except:
        return None


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