# It detects the shade of Image
# Pruple-Blue or Orange-Yellow


import cv2
import numpy as np
import os
import argparse

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


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    image = args["image"] 
    if  os.path.isfile(image): 
        print (image)
        get_shade(cv2.imread(image))

    else:
        print ("No such file found")