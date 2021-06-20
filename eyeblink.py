# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 18:22:22 2021

@author: madhur
"""
import time            
from scipy.spatial import distance as dist
import dlib
import cv2
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])                       #distance between top and bottom points
    B = dist.euclidean(eye[2],eye[4])                       #distance between top and bottom points
    C = dist.euclidean(eye[0],eye[3])                       #distance between left and right points
    asratio = (A+B)/(2*C)                                   #calculating the aspect ratio
    return asratio
        
font = cv2.FONT_HERSHEY_SIMPLEX
fontcolor = (0,255,0)
fontscale = 1
face_arr = []
cap = cv2.VideoCapture(0)        #turn on camera (since 0 means in-built camera of your laptop)
detector = dlib.get_frontal_face_detector()      #detect faces
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')     #detect face landmarks
end = time.time() + 15           #it will run for 15 secs
while time.time() < end:           #while loop for 15 secs
    res, img = cap.read()        #read camera input
    img = cv2.flip(img,1)        #flipping of the input image/frame (so that it doesn't appear like a mirror)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           #converting the input to grayscale(required for detecting faces)
    faces = detector(gray,1)                               # getting an array of face bounding box co-ordinates
    if len(faces)>len(face_arr):                           #this if block ensures for each frame the same faces aren't appended again and again for each frame
        for i in range(len(faces)-len(face_arr)):
            face_arr.append([0,0,"Fake"])
    for (i,face) in enumerate(faces):                      #this takes care of all the faces detected
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        landm = predictor(gray,face).parts()               #array of face landmarks
        '''i = 36                                          # documented this part because I got a shorter version of the same code below
        while i<41:                                        #You will understand this piece of code after seeing the face landmarks image 
            pt = landm[i]
            pt2 = landm[i+1]
            cv2.line(img,(pt.x,pt.y),(pt2.x,pt2.y),(0,255,0),1)
            i+=1
        cv2.line(img,(landm[36].x,landm[36].y),(landm[41].x,landm[41].y),(0,255,0),1)
        i = 42
        while i<47:
            pt = landm[i]
            pt2 = landm[i+1]
            cv2.line(img,(pt.x,pt.y),(pt2.x,pt2.y),(0,255,0),1)
            i+=1
        cv2.line(img,(landm[42].x,landm[42].y),(landm[47].x,landm[47].y),(0,255,0),1)'''
        cv2.rectangle(img,(x,y),(w,h),(0,255,0),2)                       #creates a rectangle according to face co-ordinates
        left = [[p.x,p.y] for p in landm[36:42]]                         #extracts left eye
        right = [[p.x,p.y] for p in landm[42:48]]                        #extracts right eye
        leftear = eye_aspect_ratio(left)                                 #calculates left eye aspect ratio
        rightear = eye_aspect_ratio(right)                               #calculates right eye aspect ratio
        ear = (leftear+rightear)/2                                       # takes an average of the eye ratios 
        #print(ear)                                                      #this is optional (only for getting the right eye aspect ratio threshold)
        if ear < 0.265:                                                  #this threshold could be different for different people
            face_arr[i][0]+=1
        else:
            if face_arr[i][0]>=2:                                        #if eye ratio is less than threshold for 2 frames the it's a blink
                face_arr[i][1]+=1
                face_arr[i][0]=0
        if face_arr[i][1]>=2:
            face_arr[i][2]="Real"                                        #if the face blinks more than twice it's a real face
        cv2.putText(img,face_arr[i][2],(x,h),font,fontscale,fontcolor,3)
        cv2.putText(img,"Blinks="+str(face_arr[i][1]),(x,y),font,fontscale,fontcolor,3)
    print(face_arr)
    cv2.imshow("frame",img)
    cv2.waitKey(1)
#print(img.shape)
#print(type(img),type(rev_im))
#print(landm.parts())
cap.release()
cv2.destroyAllWindows()
