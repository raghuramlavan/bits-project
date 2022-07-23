import cv2
import os
import sys
import traceback # for debugging
import numpy as np
import csv


from ai_component import *
from get_bb import *

# define thres in one place
def our_thres(img):
    return cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)


image_folder = "/home/lavan/conservation_drones_train_real/TrainReal/images/0000000010_0000000000"
#image_folder =  "/home/lavan/ALL_IN_ONE_RGB_IMG_ANOT"
video_name = 'orginal_second_dataset.mp4'

f = [str(i).zfill(10) for i in range(1,1730)]
pre = "0000000010_0000000000_"
#pre=""
images = [pre+n+".jpg" for n in f ]
print(images[0])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width,height), True)
#video_org = cv2.VideoWriter("original.avi", 0, 30, (width,height),False)

#imageStarsCropped = cv2.bitwise_and(imageStars, mask)
for image in images:
    print(f"processing {image} ")
    f = os.path.join(image_folder, image)
    if(  os.path.isfile(f) == False ):
        print(f"warn * {f} not found")
        continue
    frame = cv2.imread( f, cv2.IMREAD_GRAYSCALE)

    
    
    try:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except:
        err = sys.exc_info()
        print(f"{traceback.print_tb(err[1])} \n {err[0]}")
        
        


cv2.destroyAllWindows()
video.release()
print("Videowriten")