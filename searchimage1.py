# file by Anup Saxena

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# import argparse
from glob import glob

template_path_dir = "/home/divesh/Desktop/imagesearch/images/templates/*.*"
image_path_dir = "/home/divesh/Desktop/imagesearch/images/*.*"

if not os.path.exists('output'):
    os.makedirs('output')


def pattern_match(image_path, template_path):

    img_rgb = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7,7), 0)

    img_gray = cv2.Canny(blurred_image, 20, 80)
    # cv2.imshow("Canny with low thresholds", img_gray)
    # cv2.waitKey(0)
    template = cv2.imread(template_path,0)

    # cv2.imshow("image-tem", template)
    # cv2.waitKey(0)
    #gray_image2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    blurred_image2 = cv2.GaussianBlur(template, (7,7), 0)
    # cv2.imshow("image-tem", blurred_image2)
    # cv2.waitKey(0)
    img_gray2 = cv2.Canny(blurred_image2, 20, 80)
    # cv2.imshow("image-tem", img_gray2)
    # cv2.waitKey(0)

    w, h = img_gray2.shape[::-1]
    res = cv2.matchTemplate(img_gray,img_gray2,cv2.TM_CCOEFF_NORMED)
    # cv2.imshow("image-tem", res)
    # cv2.waitKey(0)
    threshold = 0.6
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


    img_base = os.path.basename(image_path)
    img_file = os.path.splitext(img_base)[0]
    tmpl_base = os.path.basename(template_path)
    tmpl_file = os.path.splitext(tmpl_base)[0]

    path_to_save = "output/%s-%s.jpg" % (tmpl_file,img_file)
    cv2.imwrite(path_to_save,img_rgb)



if __name__ == "__main__":

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", required=True, help="path to the images directory")
    # ap.add_argument("-t", "--templates", required=True, help="path to the templates directory")
    # args = vars(ap.parse_args())
    
    # image_path_dir = args["image"] 
    # template_path_dir = args["templates"] 

    for template_path in glob(template_path_dir):   
        for image_path in glob(image_path_dir):
            print (image_path, template_path )
            pattern_match(image_path, template_path)
