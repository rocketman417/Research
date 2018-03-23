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
def cont_shape_to_np(control_shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range (0, 68):
        coords[i] = (control_shape.part(i).x, control_shape.part(i).y)

    return coords

def exp_shape_to_np(exp_shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range (0, 68):
        coords[i] = (exp_shape.part(i).x, exp_shape.part(i).y)

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






#####Begin Control Picture Process#####

print("Taking control picture in \n3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

camera.capture(rawCapture, format='bgr')
control_image = rawCapture.array

print("Picture taken...")
control_image = imutils.resize(control_image, width=500)
control_gray = cv2.cvtColor(control_image, cv2.COLOR_BGR2GRAY)
control_histogram = cv2.equalizeHist(control_gray)

#saves "face box" as rects
control_rects = detector(control_histogram, 1)

#apply feature predictor to image
for (i, control_rects) in enumerate(control_rects):
    control_shape = predictor(control_histogram, control_rects)
    control_shape = cont_shape_to_np(control_shape)

    (x,y,w,h) = rect_to_bb(control_rects)
    cv2.rectangle(control_image, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(control_image, "Face #{}" .format(i+1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x,y) in control_shape:
        cv2.circle(control_image, (x,y), 1, (0,0,255), -1)
 
cv2.imshow("Marked Features", control_image)

#uncomment to show full coordinate list
#print (control_shape)

#splits the x and y values up, since the "control_shape" object does not store them as a list and they must be modified
cont_x = control_shape[:,0]
cont_y = control_shape[:,1]

#converts x and y values to lists
cont_shape_x = list(cont_x)
cont_shape_y = list(cont_y)

#uncomment to display a replot of the coordinates
#plt.scatter(cont_shape_x, cont_shape_y)
#plt.gca().invert_yaxis()
#plt.show()


#For reference: (relative to image subject) mouthR=48, mouthL=54, bottomlipT=66, toplipB=62
cont_mouthR_y = cont_shape_y[48]
cont_mouthL_y = cont_shape_y[54]
cont_bottomlipT = cont_shape_y[66]
cont_toplipB = cont_shape_y[62]
cont_mouthR_x = cont_shape_x[48]
cont_mouthL_x = cont_shape_x[54]

cont_corners_dist = cont_mouthL_x - cont_mouthR_x

cont_corners_height = (cont_mouthR_y + cont_mouthL_y)/2
cont_lips = (cont_bottomlipT + cont_toplipB)/2
cont_mouth_height_range = cont_lips - cont_corners_height

cont_lip_vert_dist = cont_bottomlipT - cont_toplipB


rawCapture.truncate(0)



#####Begin Experimental Picture Process#####

print('Taking experimental picture in \n3...')
time.sleep(1)
print('2...')
time.sleep(1)
print('1...')
time.sleep(1)


camera.capture(rawCapture, format='bgr')
exp_image = rawCapture.array
print("Picture taken...")
exp_image = imutils.resize(exp_image, width=500)
exp_gray = cv2.cvtColor(exp_image, cv2.COLOR_BGR2GRAY)
exp_histogram = cv2.equalizeHist(exp_gray)

exp_rects = detector(exp_histogram, 1)

for (i, exp_rects) in enumerate(exp_rects):
    exp_shape = predictor(exp_histogram, exp_rects)
    exp_shape = exp_shape_to_np(exp_shape)

    (x,y,w,h) = rect_to_bb(exp_rects)
    cv2.rectangle(exp_image, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(exp_image, "Face #{}" .format(i+1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x,y) in exp_shape:
        cv2.circle(exp_image, (x,y), 1, (0,0,255), -1)

user_input = raw_input('Do you want to save the experimental image as "exp_image_xxx.jpg"? (y/n)\n')

if user_input == "y":
    i = 1
    while i < 10:
        if os.path.isfile("exp_image_00" + str(i) + ".jpg") == True:
            i += 1
        elif os.path.isfile("exp_image_00" + str(i) + ".jpg") == False:
            cv2.imwrite('exp_image_00' + str(i) + '.jpg', exp_image)
            print('image saved as "exp_img_00' + str(i) + '.jpg"\n')
            break

    while 9 < i < 100:
        if os.path.isfile("exp_image_0" + str(i) + ".jpg") == True:
            i += 1
        elif os.path.isfile("exp_image_0" + str(i) + ".jpg") == False:
            cv2.imwrite('exp_image_0' + str(i) + '.jpg', exp_image)
            print('Image saved as "exp_img_0' + str(i) + '.jpg"')
            break

    if i > 99:
        print ('Error: all numbers have been exhausted; unable to write "exp_image_xxx.jpg"')

elif user_input == "n":
    print("Experimental image was not saved\n")

exp_x = exp_shape[:,0]
exp_y = exp_shape[:,1]

exp_shape_x = list(exp_x)
exp_shape_y = list(exp_y)

exp_mouthR_y = exp_shape_y[48]
exp_mouthL_y = exp_shape_y[54]
exp_bottomlipT = exp_shape_y[66]
exp_toplipB = exp_shape_y[62]
exp_mouthR_x = exp_shape_x[48]
exp_mouthL_x = exp_shape_x[54]


exp_corners_dist = exp_mouthL_x - exp_mouthR_x
#print ('Experimental horizontal corner distance: ' + str(exp_corners_dist)) #Larger number indicates smiling

exp_corners_height = (exp_mouthR_y + exp_mouthL_y)/2
exp_lips = (exp_bottomlipT + exp_toplipB)/2
exp_mouth_height_range = exp_lips - exp_corners_height
#print ('Experimental mouth height range: ' + str(exp_mouth_height_range)) #Negative number indicates not smiling, larger means smiling

exp_lip_vert_dist = exp_bottomlipT - exp_toplipB
#print('Experimental lip vertical distance: ' + str(exp_lip_vert_dist)) #Larger number indicates gap between lips, could indicate smiling



#####Compare Control with Experimental#####
dif_horiz_corner_dist = exp_corners_dist - cont_corners_dist
dif_mouth_height_range = exp_mouth_height_range - cont_mouth_height_range
dif_lip_vert_dist = exp_lip_vert_dist - cont_lip_vert_dist
#print('Difference in horizontal corner distance: ' + str(dif_horiz_corner_dist)) #positive numbers indicate happiness
#print('Difference in mouth height range: ' + str(dif_mouth_height_range)) #^^^^^
#print('Difference in lip vertical distance: ' + str(dif_lip_vert_dist))#^^^^^

#Prevents an error if this value is zero
if cont_lip_vert_dist == 0:
    cont_lip_vert_dist += 1
    
#converts the change in distance to a percent change value    
percent_change_corner_dist = (dif_horiz_corner_dist / float(abs(cont_corners_dist))) * 100
percent_change_mouth_height_range = (dif_mouth_height_range / float(abs(cont_mouth_height_range))) * 100
percent_change_lip_vert_dist = (dif_lip_vert_dist / float(abs(cont_lip_vert_dist))) * 100

print('The distance between the corners of the mouth increased by' + str(percent_change_corner_dist) + '%')
print('The distance between the top of the corners of the mouth and the bottom of the lips increased by' + str(percent_change_mouth_height_range) + '%')
print('The distance between the lips increased by' + str(percent_change_lip_vert_dist) + '%')

#These are the "rules" for what constitutes a smile. Each one that is met adds 1 to i. The final value of i determines if the person is smiling.
i=0
if percent_change_corner_dist > 25:
    i+=1

if percent_change_mouth_height_range > 45:
    i+=1

if percent_change_lip_vert_dist > 450:
    i+=1

cv2.putText(exp_image, 'Smile Probability: ' + str(i), (330, 17), 0, .5, (0, 255, 0), 2)
cv2.imshow('Final Image', exp_image)
if i==3:
    print ("This person is almost definitely smiling.")
elif i==2:
    print ("This person is likely smiling")
elif i==1:
    print ("This person is likely not smiling")
elif i==0:
    print ("This person is almost definitely not smiling")
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()








