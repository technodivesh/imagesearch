# main file to run


import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import imutils
from glob import glob


image_path_dir = "/home/divesh/Desktop/imagesearch/images/*.*"

THRESHOLD_BLUE = 75 
THRESHOLD_GREEN = 75

def get_shade(image):


    shade = ""    
    org_img = image
    # org_img = cv2.imread(image)
    # cv2.imshow("Original Image", org_img)
    # cv2.waitKey(0)

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

    # Draw biggest contour on image
    # img_cnt = cv2.drawContours(org_img.copy(), [big_c], -1, (0,255,0), thickness=2)
    # cv2.imshow("Biggest Countour", img_cnt)
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
        return 'puple-blue'
    elif ( B < THRESHOLD_BLUE and G > THRESHOLD_GREEN):
        print ("Shade is Orange Yellow")
        return 'orange-yellow'

    else:
        print ("Shade not identified---B,G,R,A ",B,G,R,A )
        return 'not-detected'


def find_spot(image_path, shade=None):

    image_obj = cv2.imread(image_path)
    shade = get_shade(image_obj)

    # print('shade--', shade) 
    # shade = "orange-yellow"    
    # shade = "puple-blue"
    # read the image 
    org_img = image_obj
    # print (org_img)
    cv2.imshow("Original Image", org_img)
    cv2.waitKey(0)

    # HSV scale conversion 
    # hsv_img = cv2.cvtColor(org_img.copy(), cv2.COLOR_BGR2HSV)
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
        # cv2.imshow("roi_mask Countour", roi_mask)
        # cv2.waitKey(0)

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

        
        # print(roi_hsv_img.shape)
        arr_zero = np.zeros(roi_hsv_img.shape)
        # print(arr_zero.shape)
        # np.place(roi_hsv_img[:,:,1], roi_hsv_img[:,:,1] == 0  , 255)
        # np.place(arr_zero[:,:,1], roi_hsv_img[:,:,1]<50  , 255)

        # Where Green < 50 and Red > 200
        arr_zero[( roi_hsv_img[:,:,1] < 150 )&( roi_hsv_img[:,:,2] > 200)] = 255

        cv2.imshow("arr_zero Image1", arr_zero)
        cv2.waitKey(0)

        # save image and read again to make it in image format
        cv2.imwrite('temp.jpg',arr_zero)
        arr_zero = cv2.imread('temp.jpg')

        # Get contour 
        arr_zero_gray = cv2.cvtColor(arr_zero.copy(), cv2.COLOR_BGR2GRAY)
        #arr_zero_blurred = cv2.GaussianBlur(arr_zero_gray.copy(), (5,5), 0)
        arr_zero_thresh = cv2.threshold(arr_zero_gray.copy(), 80, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("arr_zero_thresh Image", arr_zero_thresh)
        # cv2.waitKey(0)

        # print("thresh.shape--", thresh.shape)
        # Get contours
        _, contours,_ = cv2.findContours(arr_zero_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

        if len(contours) > 0:
            print("----This image has spots------") 
        else:
            print("----No Spot Found------") 

        final_image = org_img.copy()
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(final_image ,(x,y),(x+w,y+h),(0,255,0),thickness=2)

        cv2.imshow("final_image", final_image)
        cv2.waitKey(0)  

        img_base = os.path.basename(image_path)
        img_file = os.path.splitext(img_base)[0]

        path_to_save = "output/%s.jpg" % (img_file)
        cv2.imwrite(path_to_save,np.hstack([org_img,roi_img,final_image]))





    # Shade is 'purple-blue'
    elif shade == "puple-blue":
        thresh = cv2.threshold(blurred.copy(), 80, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow("thresh ", thresh)
        # cv2.waitKey(0)


        # Get contours
        _, contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        org_img_m = org_img.copy()[:,:,0].astype('uint8').shape
        roi_mask = np.zeros( org_img_m, np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cv2.drawContours(roi_mask, [cnt], -1, (255,255,255), thickness=-1)
        # cv2.imshow("roi_mask Countour", roi_mask)
        # cv2.waitKey(0)

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
        # print(arr_zero.shape)
        # np.place(roi_img[:,:,1], roi_img[:,:,1] == 0  , 255)
        # np.place(roi_img[:,:,0], roi_img[:,:,0] == 0  , 255)
        # np.place(roi_img[:,:,1], roi_img[:,:,1] == 0  , 0)
        # np.place(arr_zero[:,:,1], roi_img[:,:,1]<50  , 255)

        # Blue < 50 and Green > 100  
        arr_zero[(roi_img[:,:,0] < 50) & (roi_img[:,:,1] > 100) | 
            (roi_img[:,:,0] > 200 ) & 
            (roi_img[:,:,1] > 200 ) & 
            (roi_img[:,:,2] > 200 ) ] = 255


        # np.place(arr_zero[:,:,0], roi_img[:,:,0]<50  , 255)
        # np.place(arr_zero[:,:,1], roi_img[:,:,1]>100  , 255)



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
        else:
            print("----No Spot Found------") 

        final_image = org_img.copy()
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(final_image ,(x,y),(x+w,y+h),(0,255,0),thickness=2)

        cv2.imshow("final_image", final_image)
        cv2.waitKey(0)    

        img_base = os.path.basename(image_path)
        img_file = os.path.splitext(img_base)[0]

        path_to_save = "output/%s.jpg" % (img_file)
        cv2.imwrite(path_to_save,np.hstack([org_img,roi_img,final_image]))

    ## delete only if file exists ##
    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    image_path = args["image"] 
    if  os.path.isfile(image_path): 
        print (image_path)
        find_spot(image_path)

    else:
        print ("No such file found")


    # for image_path in glob(image_path_dir):
    #         print (image_path)
    #         find_spot(image_path)