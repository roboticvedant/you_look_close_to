import os
import cv2 as cv
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR,'img')
current_id=0
label_ids = {}

x_train = []
y_labels = []

haar_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()

for root, dir, files in os.walk(IMG_PATH):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(path)
            img_load = cv.imread(path)
            gray = cv.cvtColor(img_load,cv.COLOR_BGR2GRAY)
            if not label in label_ids:
                label_ids[label]=current_id
                #print(label_ids[label])
                current_id+=1
            id_ = label_ids[label]
            print(id_)

            face = haar_cascade.detectMultiScale(gray,1.5,2)
            for(x,y,w,h) in face:
                roi = gray[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")