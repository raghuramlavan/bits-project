import cv2
import os
import sys
import traceback # for debugging
import numpy as np



from ai_component import *
from get_bb import *

# define thres in one place
def our_thres(img):
    return cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)



image_folder =  "/home/lavan/ALL_IN_ONE_RGB_IMG_ANOT"
video_name = 'cc.avi'

f = [str(i).zfill(3) for i in range(1,1730)]

images = [n+".jpg" for n in f ]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height),False)
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
print(feat_buff_centroid)
feat_buff_full = False # when the programm is being started

for image in images:

    frame = cv2.imread(os.path.join(image_folder, image) , cv2.IMREAD_GRAYSCALE)

    
    
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
            output = clean.copy()
            for cen in feat_buff_centroid[feat_buff_ptr]:
                for (x,y),(x2,y2) in zip(L,R):
                    #print(f"{cen[0]} {x2}")
                    if(cen[0]>x and cen[0] < x2 and cen [1]>y and cen[1]<y2):
                        feat_buff_lable[feat_buff_ptr][ii] = 1
                        continue
                    else : 
                        feat_buff_lable[feat_buff_ptr][ii] = 0
                #print(f"{len(feat_buff_lable[feat_buff_ptr])} {ii} {len(feat_buff_centroid[feat_buff_ptr])}")
                if(feat_buff_lable[feat_buff_ptr][ii] == 1): 
	                cv2.rectangle(output, (x, y), (x2, y2), (0, 255, 0), 3)
	                cv2.circle(output, (int(cen[0]), int(cen[1])), 40, (0, 0, 255), -1)
	                print(f"doing {x} {y} {x2} {y2} {cen[0]} {cen[1]} {output.shape}")
                ii = ii + 1
            cv2.imshow("test",clean)
            k = cv2.waitKey(0) # 0==wait forever

            cv2.rectangle(output, (10, 10), (100, 100), (0, 255, 0), 3)
            video.write(output)
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
        exit()
        


cv2.destroyAllWindows()
video.release()