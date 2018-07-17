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

THRESHOLD_BLUE = 75 
THRESHOLD_GREEN = 75
# Assuming in puple-blue shade BLUE > 75 and GREEN < 75
# ans in orange-yellow shade BLUE < 75 and GREEN > 75

# img_rgb = cv2.imread('images/20180630_154201.jpg')


def find_roi(image,filename):

    shade = ""    
    # read the image 
    org_img = cv2.imread(image)
    # print (org_img)
    cv2.imshow("Original Image", org_img)
    cv2.waitKey(0)

    # # To sharpen the image
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharp_img = cv2.filter2D(org_img.copy(), -1, kernel)
    # cv2.imshow("Sharpen Image", sharp_img)
    # cv2.waitKey(0)

    # HSV scale conversion 
    hsv_img = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv_img Image", hsv_img)
    cv2.waitKey(0)

    # Gray scale conversion
    gray = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)

    # Blur the image
    blurred = cv2.GaussianBlur(gray.copy(), (55,5), 0)
    # cv2.imshow("Blurred Image", blurred)
    # cv2.waitKey(0)

    thresh = cv2.threshold(blurred.copy(), 120, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("thresh Image0", thresh)
    # cv2.waitKey(0)




    # # Threshholding to convert it in binary image  # 120 For yellow
    # thresh = cv2.threshold(blurred.copy(), 120, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("thresh Image", thresh)
    # cv2.waitKey(0)

    # Get contours
    _, contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    org_img_m = org_img.copy()[:,:,0].astype('uint8').shape
    roi_mask = np.zeros( org_img_m, np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(roi_mask, [cnt], -1, (255,255,255), thickness=-1)
    cv2.imshow("roi_mask Countour", roi_mask)
    cv2.waitKey(0)


    # Morphological tranformation and dilation 
    kernal = np.ones((5,5), "uint8")

    roi_mask=cv2.dilate(roi_mask, kernal)
    roi_img=cv2.bitwise_and(org_img,org_img, mask=roi_mask)
    # cv2.imshow("roi_img", roi_img)
    # cv2.waitKey(0)

    # Convert to HSV
    roi_hsv_img = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2HSV)
    cv2.imshow("roi_hsv_img Image", roi_hsv_img)
    cv2.waitKey(0)


    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(roi_hsv_img, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(roi_hsv_img, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # set my output img to zero everywhere except my mask
    output_img = roi_hsv_img.copy()
    output_img[np.where(mask==0)] = 0

    cv2.imshow("output_img Image1", output_img)
    cv2.waitKey(0)


    # set my output img to zero everywhere except my mask
    output_img = org_img.copy()
    output_img[np.where(mask==0)] = 0

    cv2.imshow("output_img Image2", output_img)
    cv2.waitKey(0)




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