# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 18:22:22 2021

@author: madhu
"""
import time
from scipy.spatial import distance as dist
import dlib
import cv2
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    asratio = (A+B)/(2*C)
    return asratio
        
font = cv2.FONT_HERSHEY_SIMPLEX
fontcolor = (0,255,0)
fontscale = 1
face_arr = []
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
end = time.time() + 15
while time.time() < end:
    res, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray,1)
    if len(faces)>len(face_arr):
        for i in range(len(faces)-len(face_arr)):
            face_arr.append([0,0,"Fake"])
    for (i,face) in enumerate(faces):
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        landm = predictor(gray,face).parts()
        '''i = 36
        while i<41:
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
        cv2.rectangle(img,(x,y),(w,h),(0,255,0),2)
        left = [[p.x,p.y] for p in landm[36:42]]
        right = [[p.x,p.y] for p in landm[42:48]]
        leftear = eye_aspect_ratio(left)
        rightear = eye_aspect_ratio(right)
        ear = (leftear+rightear)/2
        print(ear)
        if ear < 0.265:
            face_arr[i][0]+=1
        else:
            if face_arr[i][0]>=2:
                face_arr[i][1]+=1
                face_arr[i][0]=0
        if face_arr[i][1]>=2:
            face_arr[i][2]="Real"
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