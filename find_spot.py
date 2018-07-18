# main file to run


import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import imutils
import detect_shade


def find_spot(image, shade=None):

    print('shade--', shade) 
    # shade = "orange-yellow"    
    # shade = "puple-blue"
    # read the image 
    org_img = image
    # print (org_img)
    cv2.imshow("Original Image", org_img)
    cv2.waitKey(0)

    # HSV scale conversion 
    hsv_img = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv_img Image", hsv_img)
    # cv2.waitKey(0)

    # Gray scale conversion
    gray = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)

    # Blur the image
    blurred = cv2.GaussianBlur(gray.copy(), (55,5), 0)
    # cv2.imshow("Blurred Image", blurred)
    # cv2.waitKey(0)


    if shade == 'orange-yellow':
        thresh = cv2.threshold(blurred.copy(), 120, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("thresh ", thresh)
        cv2.waitKey(0)

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


        # cv2.imshow("arr_zero Image1", arr_zero)
        # cv2.waitKey(0)

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

        if len(contours) > 0:
            print("----This image has spots------") 

        final_image = org_img.copy()
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(final_image ,(x,y),(x+w,y+h),(0,0,255),thickness=2)

        cv2.imshow("final_image", final_image)
        cv2.waitKey(0)   

    # Shade is 'purple-yellow'
    elif shade == "puple-blue":
        thresh = cv2.threshold(blurred.copy(), 80, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("thresh ", thresh)
        cv2.waitKey(0)


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
        # roi_hsv_img = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2HSV)
        # cv2.imshow("roi_hsv_img Image", roi_hsv_img)
        # cv2.waitKey(0)

        
        # print(roi_hsv_img.shape)
        arr_zero = np.zeros(roi_img.shape)
        print(arr_zero.shape)
        # Green > 100 and Blue < 50
        # np.place(roi_img[:,:,1], roi_img[:,:,1] == 0  , 255)
        np.place(roi_img[:,:,1], roi_img[:,:,1] == 0  , 0)
        np.place(roi_img[:,:,0], roi_img[:,:,0] == 0  , 255)

        # np.place(arr_zero[:,:,1], roi_img[:,:,1]<50  , 255)
        np.place(arr_zero[:,:,1], roi_img[:,:,1]>100  , 255)
        np.place(arr_zero[:,:,0], roi_img[:,:,0]<50  , 255)


        # cv2.imshow("arr_zero Image1", arr_zero)
        # cv2.waitKey(0)

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

        if len(contours) > 0:
            print("----This image has spots------") 

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
        print (image)
        image_obj = cv2.imread(image)
        shade = detect_shade.get_shade(image_obj)
        find_spot(image_obj,shade)

    else:
        print ("No such file found")