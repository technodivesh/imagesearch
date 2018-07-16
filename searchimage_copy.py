# file by Anup Saxena

import cv2
import numpy as np
from matplotlib import pyplot as plt



img_rgb = cv2.imread('images/20180630_154201.jpg')
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (7,7), 0)

img_gray = cv2.Canny(blurred_image, 20, 80)
cv2.imshow("Canny with low thresholds", img_gray)
cv2.waitKey(0)
template = cv2.imread('images/template3.jpg',0)

cv2.imshow("image-tem", template)
cv2.waitKey(0)
#gray_image2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
blurred_image2 = cv2.GaussianBlur(template, (7,7), 0)
cv2.imshow("image-tem", blurred_image2)
cv2.waitKey(0)
img_gray2 = cv2.Canny(blurred_image2, 20, 80)
cv2.imshow("image-tem", img_gray2)
cv2.waitKey(0)

w, h = img_gray2.shape[::-1]
res = cv2.matchTemplate(img_gray,img_gray2,cv2.TM_CCOEFF_NORMED)
cv2.imshow("image-tem", res)
cv2.waitKey(0)
threshold = 0.6
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('images2/res.png',img_rgb)
