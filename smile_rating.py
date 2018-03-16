import numpy as np
import dlib
import cv2
import imutils
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import os
import matplotlib.pyplot as plt

#functions to make values compatible
#this converts the dlib feature prediction to np values
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range (0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

#this converts the dlib rectangle to numerical values to be used by cv2
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x,y,w,h)
             
#define face detector and feature predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#take a picture and convert to grayscale, then equalize histogram
camera=PiCamera()
rawCapture = PiRGBArray(camera)

print("taking picture in \n3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

camera.capture(rawCapture, format='bgr')
image = rawCapture.array
print("Picture taken...")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.equalizeHist(gray)

#saves "face box" as rects
rects = detector(histogram, 1)

#apply feature predictor to image
for (i, rects) in enumerate(rects):
    shape = predictor(histogram, rects)
    shape = shape_to_np(shape)

    (x,y,w,h) = rect_to_bb(rects)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(image, "Face #{}" .format(i+1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x,y) in shape:
        cv2.circle(image, (x,y), 1, (0,0,255), -1)

cv2.imshow("Marked Features", image)

#ask for user input to trigger naming logic and save image
user_input = raw_input('Do you want to save the final image as "feat_image_xxx.jpg"? (y/n)\n')

if user_input == "y":
    i = 1
    while i < 10:
        if os.path.isfile("feat_image_00" + str(i) + ".jpg") == True:
            i += 1
        elif os.path.isfile("feat_image_00" + str(i) + ".jpg") == False:
            cv2.imwrite('feat_image_00' + str(i) + '.jpg', image)
            print('image saved as "feat_img_00' + str(i) + '.jpg"\n')
            break

    while 9 < i < 100:
        if os.path.isfile("feat_image_0" + str(i) + ".jpg") == True:
            i += 1
        elif os.path.isfile("feat_image_0" + str(i) + ".jpg") == False:
            cv2.imwrite('feat_image_0' + str(i) + '.jpg', image)
            print('image saved as "feat_img_0' + str(i) + '.jpg"')
            break

    if i > 99:
        print ('error: all numbers have been exhausted; unable to write "feat_image_xxx.jpg"')

elif user_input == "n":
    print("image was not saved\n")

#uncomment to show full coordinate list
#print (shape)

#splits the x and y values up, since the "shape" object does not store them as a list
x = shape[:,0]
y = shape[:,1]

#converts x and y values to lists
shape_x = list(x)
shape_y = list(y)

#uncomment to display a replot of the coordinates
#plt.scatter(shape_x, shape_y)
#plt.gca().invert_yaxis()
#plt.show()


#For reference: (relative to image) mouthR=48, mouthL=54, bottomlipT=66, toplipB=62
mouthR = shape_y[48]
mouthL = shape_y[54]
bottomlipT = shape_y[66]
toplipB = shape_y[62]

corners = (mouthR + mouthL)/2
lips = (bottomlipT + toplipB)/2

#this is the value for happiness seen just in the mouth. A larger (positive) number indicates happiness.
mouth_rating = lips - corners
print ('Smile Rating: ' + str(mouth_rating))

cv2.waitKey(0)
cv2.destroyAllWindows()
