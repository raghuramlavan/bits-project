import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


dir = "/home/lavan/ALL_IN_ONE_RGB_IMG_ANOT"
if(len(sys.argv)>1):
    dir = sys.argv[1] 
else:
    dir = "/home/lavan/ALL_IN_ONE_RGB_IMG_ANOT"
img = cv2.imread(f"{dir}/001.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()