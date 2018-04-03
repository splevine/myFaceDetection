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

def detect_faces(f_cascade, colored_img, scaleFactor, minNeighbors):
    img_copy = np.copy(colored_img)
    # convert the test image to gray image - openCV face detector expets gray images
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # showImage('Test 1', gray_img)

    # detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors);

    # print the number of faces found
    print('Faces found: ', len(faces))

    # go over faces and draw rectangle on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy

test2 = cv2.imread('data/test3.jpg')

face_detected_img = detect_faces(haar_face_cascade, test2, 1.2, 5)

showImage('Color Test 2', face_detected_img)





