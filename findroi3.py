# pip install opencv-python
# pip install matplotlib
# sudo apt-get install python3.6-tk


import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import imutils
from copy import deepcopy

# img_rgb = cv2.imread('images/20180630_154201.jpg')


def find_roi(image,filename):
    
    # read the image 
    org_img = cv2.imread(image)
    print (org_img)
    cv2.imshow("Original Image", org_img)
    cv2.waitKey(0)

    # Gray scale conversion
    gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.waitKey(0)

    # Blur the image
    blurred = cv2.GaussianBlur(gray, (55,5), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)


    # Threshholding to convert it in binary image
    thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("thresh Image", thresh)
    cv2.waitKey(0)

    # Get contours
    _, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    #find the biggest area contour
    big_c = max(contours, key=cv2.contourArea)
    print(big_c)

    # Draw biggest contour on image
    img_cnt = cv2.drawContours(org_img, [big_c], -1, (0,255,0), thickness=2)
    cv2.imshow("Biggest Countour", img_cnt)
    cv2.waitKey(0)

    rect = cv2.minAreaRect(big_c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    minRect = cv2.drawContours(org_img, [box], -1, (0, 0, 255),thickness=2 )
    cv2.imshow("minRect Countour", minRect)
    cv2.waitKey(0)


    mean_val_cont = cv2.mean(org_img, mask = None)
    print ("mean_val of skin -------------------------BGR", mean_val_cont)


    mask = np.zeros(gray.shape,np.uint8)
    print (mask)



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