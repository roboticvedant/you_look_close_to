import cv2 as cv #computer vison
import pickle # id extraction aid
#
cap = cv.VideoCapture(0) #web cam stream
#
haar_cascade=cv.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') # haar cascade model

recognizer = cv.face.LBPHFaceRecognizer_create() # initiate face recognizer
recognizer.read('trainer.yml') # calls trained model

labels = {} # blank dictionary  to accomodate labels

with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels ={v:k for k,v in og_labels.items()} # invers order of label

while True: # main loop
    ret, frame = cap.read()
    cv.imshow("Output", frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray, 1.1, 10)
    for(x,y,w,h) in face_rect:
        box = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        roi_gray = gray[y:y+h,x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 70:
            print(labels[id_])
            cv.putText(frame,labels[id_],(x,y),cv.FONT_HERSHEY_TRIPLEX,1 , (0,255,0))

            cv.imshow("Output",frame)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

