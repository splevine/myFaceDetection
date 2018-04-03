#import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
# %matplotlib inline

# Images loaded as BGR
# Helper function for conversion to RGB

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def showImage(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## Haar cascade classifier
# XML training files for Haar cascade are stored in 'opencv/data/haarcascadesfolder'

# load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

#load test image
test1 = cv2.imread('data/test1.jpg')

# convert the test image to gray image - openCV face detector expets gray images
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

# display gray image using matplotlib
plt.show()

#display the gray image using OpenCV
# cv2.imshow('Test Imag', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# showImage('Test 1', gray_img)

# detect multiscale (some images may be closer to camera than others) images
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

# print the number of faces found
print('Faces found: ', len(faces))

# go over faces and draw rectangle on original colored img
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x,y), (x+w, y+h), (0, 255, 0), 2)

showImage('Color Test 1', test1)





