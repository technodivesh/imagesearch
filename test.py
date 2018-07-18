# pip install opencv-python
# pip install matplotlib
# sudo apt-get install python3.6-tk

from PIL import Image

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
    cv2.imshow("roi_img", roi_img)
    cv2.waitKey(0)

    # Convert to HSV
    roi_hsv_img = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2HSV)
    cv2.imshow("roi_hsv_img Image", roi_hsv_img)
    cv2.waitKey(0)

    
    # print(roi_hsv_img.shape)
    arr_zero = np.zeros(roi_hsv_img.shape)
    print(arr_zero.shape)
    np.place(roi_hsv_img[:,:,1], roi_hsv_img[:,:,1] == 0  , 255)
    np.place(arr_zero[:,:,1], roi_hsv_img[:,:,1]<50  , 255)


#     arr_zero = Image.fromarray(arr_zero, 'RGB')

    cv2.imshow("arr_zero Image1", arr_zero)
    cv2.waitKey(0)

    # save image and read again to make it in image format
    cv2.imwrite('temp.jpg',arr_zero)
    arr_zero = cv2.imread('temp.jpg')
    
    # Get contour 
    arr_zero_gray = cv2.cvtColor(arr_zero.copy(), cv2.COLOR_BGR2GRAY)
    #arr_zero_blurred = cv2.GaussianBlur(arr_zero_gray.copy(), (5,5), 0)
    arr_zero_thresh = cv2.threshold(arr_zero_gray.copy(), 80, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("arr_zero_thresh Image", arr_zero_thresh)
    cv2.waitKey(0)

    # print("thresh.shape--", thresh.shape)
    # Get contours
    _, contours,_ = cv2.findContours(arr_zero_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    print (len(contours))

    final_image = org_img.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(final_image ,(x,y),(x+w,y+h),(0,0,255),thickness=2)

    cv2.imshow("final_image", final_image)
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