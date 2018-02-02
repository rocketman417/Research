import numpy as np
import argparse
import dlib
import cv2
import imutils
import os

#functions to convert dlib's info to be compatible with cv2
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

    return (x,y,w,h)y

#define input image argument
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args=vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#take input image, convert to grayscale, then equalize histogram
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.equalizeHist(gray)

#saves "face box" as rects
rects = detector(histogram, 1)

#apply feature predictor to image
for (i, rects) in enumerate(rects):
    shape = predictor(histogram, rects)
    shape = shape_to_np(shape)

    (x,y,w,h) = rect_to_bb(rects )
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(image, "Face #{}" .format(i+1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x,y) in shape:
        cv2.circle(image, (x,y), 1, (0,0,255), -1)

cv2.imshow("Marked Features", image)

#ask for user input to trigger naming logic and save image
user_input = raw_input('Do you want to save the final image as "feat_image_xxx.jpg"? (y/n)')

if user_input == "y":
    i = 1
    while i < 10:
        if os.path.isfile("feat_image_00" + str(i) + ".jpg") == True:
            i += 1
        elif os.path.isfile("feat_image_00" + str(i) + ".jpg") == False:
            cv2.imwrite('feat_image_00' + str(i) + '.jpg', image)
            print('image saved as "feat_img_00' + str(i) + '.jpg"')
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
    print("image was not saved")
    
cv2.waitKey(0)
cv2.destroyAllWindows()
