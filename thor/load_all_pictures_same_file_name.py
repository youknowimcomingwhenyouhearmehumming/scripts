import os
import glob 
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.chdir('C:/Users/Bruger/Documents/Uni/Abu dhabi/data/ballpictures_jesper/TestDrone640_2')
    
for filename in glob.glob("*.png"): # This line take all the files of the filename .png from the current folder. Source http://stackoverflow.com/questions/6997419/how-to-create-a-loop-to-read-several-images-in-a-python-script
    col=Image.open(filename)
    gray = col.convert('L') #Here and the next line, the picture are turned into a white/black format since 
    #it then is faster to analyse afterward. 
    bw = np.asarray(gray).copy()




import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('image_53.png')
imgplot = plt.imshow(img)


#
#im = cv2.imread('image_53.png')
#im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
#
#plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
#plt.show()

img = cv2.imread(filename)
cv2.imshow("Shapes", img) 








