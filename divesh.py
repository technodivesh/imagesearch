# pip install opencv-python
# pip install matplotlib
# sudo apt-get install python3.6-tk


import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

# img_rgb = cv2.imread('images/20180630_154201.jpg')


def find_roi(image,filename):
    
    # read the image 
    img = cv2.imread(image)  
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Gray scale conversion 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)

    # For pink / purple 
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh Image", thresh)
    cv2.waitKey(0)

    thresh = cv2.erode(thresh, None, iterations=5)
    cv2.imshow("Blurred Image", thresh)
    # cv2.waitKey(0)
    print (thresh)

    # thresh = cv2.dilate(thresh, None, iterations=5)
    # cv2.imshow("Blurred Image", thresh)
    # cv2.waitKey(0)


    # cv2.imshow("Original Image", thresh)
    # cv2.waitKey(0)











if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    image = args["image"] 
    if  os.path.isfile(image): 
        filename, ext = os.path.splitext(image)
        print (image, filename)
        find_roi(image, filename)

    else:
        print ("No such file found")