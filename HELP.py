# To downscale

scale_percent = 60 # percent of original size
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

