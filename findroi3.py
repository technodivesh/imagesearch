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

    # Gray scale conversion
    gray = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)

    # Blur the image
    blurred = cv2.GaussianBlur(gray.copy(), (55,5), 0)
    # cv2.imshow("Blurred Image", blurred)
    # cv2.waitKey(0)

    # Threshholding to convert it in binary image
    thresh = cv2.threshold(blurred.copy(), 80, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("thresh Image", thresh)
    # cv2.waitKey(0)

    # Get contours
    _, contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    #find the biggest area contour
    big_c = max(contours, key=cv2.contourArea)
#     print(big_c)

    # Draw biggest contour on image
    img_cnt = cv2.drawContours(org_img.copy(), [big_c], -1, (0,255,0), thickness=2)
    cv2.imshow("Biggest Countour", img_cnt)
    cv2.waitKey(0)

    # rect = cv2.minAreaRect(big_c)
    # box = cv2.boxPoints(rect)
    # # convert all coordinates floating point values to int
    # box = np.int0(box)
    # # print(box)
    # # draw a red 'nghien' rectangle
    # minRect = cv2.drawContours(org_img, [box], -1, (0, 0, 255),thickness=2 )
    # cv2.imshow("minRect Countour", minRect)
    # cv2.waitKey(0)

    org_img_m = org_img.copy()[:,:,0].astype('uint8').shape
    mask = np.zeros( org_img_m, np.uint8)
    big_c_mask = cv2.drawContours(mask, [big_c], -1, (255,255,255), thickness=-1)
    # cv2.imshow("Mask Countour", big_c_mask)
    # cv2.waitKey(0)


    B,G,R,A = cv2.mean(org_img.copy(), mask=big_c_mask)
    # print ("Mean BGR--------", B,G,R,A)

    if ( B > THRESHOLD_BLUE and G < THRESHOLD_GREEN):
        print ("Shade is Purple Blue")
        shade = 'puple-blue'
    elif ( B < THRESHOLD_BLUE and G > THRESHOLD_GREEN):
        print ("Shade is Orange Yellow")
        shade = 'orange-yellow'

    else:
        print ("Shade not identified---B,G,R,A ",B,G,R,A )
        shade = None

    # if shade and shade == 'puple-blue':
    #     # For pink / purple 
    #     thresh = cv2.threshold(blurred.copy(), 80, 255, cv2.THRESH_BINARY)[1]
    #     cv2.imshow("Thresh Spot Image", thresh)
    #     cv2.waitKey(0)

    # HSV scale conversion 
    # hsv_img = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv_img Image", hsv_img)
    # cv2.waitKey(0)

    # To sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp_img = cv2.filter2D(gray.copy(), -1, kernel)
    cv2.imshow("Sharpen Image", sharp_img)
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