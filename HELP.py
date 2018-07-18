# To downscale

scale_percent = 50 # percent of original size
width = int(blurred.shape[1] * scale_percent / 100)
height = int(blurred.shape[0] * scale_percent / 100)
dim = (width, height)
 
# resize image
resized = cv2.resize(blurred.copy(), dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


######################################################################3

# To Upscale

scale_percent = 200 # percent of original size
width = int(blurred.shape[1] * scale_percent / 100)
height = int(blurred.shape[0] * scale_percent / 100)
dim = (width, height)
 
# resize image
resized = cv2.resize(blurred.copy(), dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################################################

# To detect red color from HSV model

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


######################################################################
# Find biggest contour