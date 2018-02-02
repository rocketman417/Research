import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#take a picture
camera=PiCamera()
rawCapture = PiRGBArray(camera)

time.sleep(1)

camera.capture(rawCapture, format='bgr')
image = rawCapture.array

#cv2.imshow('image', image)
cv2.waitKey(0)

#face cascade
face_cascade = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')

#equalize histogram   
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray', gray)
histogram = cv2.equalizeHist(gray)
#cv2.imshow("histogram", histogram)
#cv2.imwrite ("histogram.jpg", histogram)
cv2.waitKey(0)

faces = face_cascade.detectMultiScale(histogram, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0 , 0), 2)
cv2.imshow('cascade_image', image)

#naming logic for cascade image
i = 1
while i < 10:
    if os.path.isfile("cas_image_00" + str(i) + ".jpg") == True:
        i += 1
    elif os.path.isfile("cas_image_00" + str(i) + ".jpg") == False:
        cv2.imwrite('cas_image_00' + str(i) + '.jpg', image)
        print('image saved as "cas_img_00' + str(i) + '.jpg"')
        break

while 9 < i < 100:
    if os.path.isfile("cas_image_0" + str(i) + ".jpg") == True:
        i += 1
    elif os.path.isfile("cas_image_0" + str(i) + ".jpg") == False:
        cv2.imwrite('cas_image_0' + str(i) + '.jpg', image)
        print('image saved as "cas_img_0' + str(i) + '.jpg"')
        break

if i > 99:
    print ('error: all numbers have been exhausted; unable to write "cas_image_xxx.jpg"')

cv2.waitKey(0)
cv2.destroyAllWindows()
