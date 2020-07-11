from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import cv2
from keras.models import load_model
import time
import cv2
import requests


#face_cascade = cv2.CascadeClassifier('C:\\Users\\Tanmoy\\Documents\\openCV project\\haarcascade_frontalface_default.xml')
new_model = load_model('new.h5')
new_model.summary()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\\Users\\Tanmoy\\Documents\\openCV project\\haarcascade_frontalface_default.xml')


while True:
    
    ret, img = cap.read()
    #cv2.imshow('cam',img)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,z,h) in faces:
        k = cv2.rectangle(img,(x,y),(x+z,y+h),(255,0,0),2)
        face_img = img[y:y+z,x:x+z]
        resized = cv2.resize(face_img,(150,150))
        #normalized = resized/255.0
        reshaped = np.reshape(resized,(1,150,150,3))
        rslt = new_model.predict(reshaped)
                        
        print(rslt)
        if rslt[0][0] == 1:
            pred = "without mask"
        else:
            pred = "with mask"

        print(pred)
    



        cv2.putText(img,pred,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('img2',resized)
    
    cv2.imshow('img',img)
        



    if cv2.waitKey(1)== 27:
        break

cv2.destroyAllWindows()
