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
    img2 = deepcopy(org_img)
    img3 = deepcopy(org_img)
    hsv_img = deepcopy(org_img)
    org_image = deepcopy(org_img)
    cv2.imshow("Original Image", org_img)
    cv2.waitKey(0)


    # Blur the image
    blurred = cv2.GaussianBlur(org_img, (55,5), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)
    # HSV scale conversion 
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv_img Image", hsv_img)
    # Gray scale conversion 
    gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.waitKey(0)


    # resized = imutils.resize(img, width=300)
    # ratio = img.shape[0] / float(resized.shape[0])
    # cv2.imshow("resized Image", resized)
    # cv2.waitKey(0)

    # Threshholding to convert it in binary image
    thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("thresh Image", thresh)
    cv2.waitKey(0)

    _, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )


    img_cnt = cv2.drawContours(img2, contours, -1, (0,255,0), thickness=2)
    cv2.imshow("img_cnt Image", img_cnt)
    cv2.waitKey(0)

    cnt = contours[1]
    epsilon = 0.05*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    img_approx = cv2.drawContours(img3, [approx], -1, (0,255,0), thickness=2)
    cv2.imshow("img_approx Image", img_approx)
    cv2.waitKey(0)


    for cnt in contours:
        # To draw a rectangle on roi
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        print("box1--",box)
        box = np.int0(box)
        print("box2--",box)
        im = cv2.drawContours(org_image,[box],0,(0,0,255),2)
    cv2.imshow("im Image", im)
    cv2.waitKey(0)

    # masked_data = cv2.bitwise_and(org_image, org_image, mask=cnt)
    # cv2.imshow("masked_data Image", masked_data)
    # cv2.waitKey(0)

    # cv2.imshow("approx Image", approx)
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