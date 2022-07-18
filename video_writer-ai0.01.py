import cv2
import os
import sys
import traceback # for debugging
import numpy as np
import csv

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras


from ai_component import *
from get_bb import *

model = keras.models.load_model('ai_detector.keras')
maximum_in_dis = 294.49
model = keras.models.load_model('ai_detector.keras')


# define thres in one place
def our_thres(img):
    return cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)



image_folder =  "/home/lavan/ALL_IN_ONE_RGB_IMG_ANOT"
video_name = 'cc_predicted.mp4'

f = [str(i).zfill(3) for i in range(1,1730)]

images = [n+".jpg" for n in f ]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width,height), True)
#video_org = cv2.VideoWriter("original.avi", 0, 30, (width,height),False)

#imageStarsCropped = cv2.bitwise_and(imageStars, mask)
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

last_frame = []
bufferlength = 5
for i in range(bufferlength):
    img = cv2.imread(os.path.join(image_folder, images[0]) , cv2.IMREAD_GRAYSCALE)
    last_frame.append(our_thres(img))
#implement circular buffer

index = 0
feat_buff_size = 20
feat_buff_centroid = [[] for i in range(feat_buff_size)]
feat_buff_lable    = [[] for i in range(feat_buff_size)]
feat_buff_human    = [[] for i in range(feat_buff_size)]
feat_buff_values   = [[] for i in range(feat_buff_size)]
feat_buff_ptr =0
feature = []
feat_buff_full = False # when the programm is being started

for image in images:
    print(f"processing {image} ")
    f = os.path.join(image_folder, image)
    if(  os.path.isfile(f) == False ):
        print(f"warn * {f} not found")
        continue
    frame = cv2.imread( f, cv2.IMREAD_GRAYSCALE)

    
    
    try:
        L,R = yolo2bb(image_folder,image,frame.shape)
        
        #ret,thresh1 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
        thresh1 = our_thres(frame)

        # cleaning buffer
        clean = thresh1
        for i in range(bufferlength):
            clean = cv2.bitwise_and(clean, last_frame[i])

        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(clean,4, cv2.CV_32S)

        feat_buff_centroid[feat_buff_ptr] = centroid
        feat_buff_values[feat_buff_ptr] = values
        if(feat_buff_ptr == feat_buff_size-1 ): feat_buff_full = True
        

        
        #print(f"{image} - > totalLables -  > {values[-1]}")
        
        #filtered = cv2.filter2D(src=thresh1, ddepth=-1, kernel=kernel1)
        
        if(feat_buff_full):
            feature = extract_features(feat_buff_centroid,feat_buff_values,feat_buff_size,feat_buff_ptr)
            
            feat_buff_lable[feat_buff_ptr] =[ 0 for i in range(len(centroid))]
            #print(f"{len(feat_buff_centroid[feat_buff_ptr])} - {len(feature)} {len(feature[0])}")
          
            #print(feature)

            ii=0
            (l,h)=clean.shape
            output = np.zeros((l,h,3),np.uint8)
            for ff in feature:
                o= model.predict(np.asarray([ x/maximum_in_dis for x in ff[0:19]] ,dtype=np.float32 ).reshape((1,19)),verbose="none")
                cen=feat_buff_centroid[feat_buff_ptr][ii]
                ii = ii+1
                if(o[0][0]>0.4): 
                    print("found HUMAN")
                    cv2.circle(output, (int(cen[0]), int(cen[1])), 4, (0, 0, 255), -1)


            video.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        # reset the circular buffer
        if( index > len(last_frame)-1 ):
            index = 0
        last_frame[index]=thresh1
        index = index + 1
        #video_org.write(frame)
        
        feat_buff_ptr = (feat_buff_ptr + 1)%feat_buff_size
    except:
        err = sys.exc_info()
        print(f"{traceback.print_tb(err[1])} \n {err[0]}")
        
        


cv2.destroyAllWindows()
video.release()
print("Videowriten")