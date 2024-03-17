import cv2
import numpy as np
#import image
path_of_image = 'FIR_images_v1/Airport PSAIRPORT-0005-2017-3336__1.jpg' #Give path of image
image = cv2.imread(path_of_image)
#cv2.imshow('orig',image)
#cv2.waitKey(0)

image = cv2.resize(image, (720, 720))

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# cv2.imshow('second',thresh)
# cv2.waitKey(0)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilated',img_dilation)
# cv2.waitKey(0)

#find contours
ctrs, im2 = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)

# print(ctrs[0].shape)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

boxes = []

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    boxes.append([(x,y), (x+w,y+h)])
    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    # cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    # cv2.waitKey(0)

# print(boxes[0][0], boxes[0][1])
for i in range(len(boxes)):
    cv2.rectangle(image,boxes[i][0], boxes[i][1],(0,255,0),1)


cv2.imshow('marked areas', image)
cv2.waitKey(0)   