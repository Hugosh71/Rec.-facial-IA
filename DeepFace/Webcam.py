import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam is disconnected or cannot be used, try again ...")

while True : 
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for (w, x, y, z) in faces : 
        cv2.rectangle(frame, (w,x), (w + y, x + z), (0,255,0),2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result[0]['dominant_emotion'], 
                (0, 50), font, 2, (0, 0, 255),
                2, cv2.LINE_4)
    
    cv2.imshow('Original video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
