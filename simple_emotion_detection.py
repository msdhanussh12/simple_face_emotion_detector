import cv2
import numpy as np

#load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    #convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #extract the region of interest (the face) for emotion detection
        face_roi = gray[y:y+h, x:x+w]
        #simple emotion detection based on the face features
        #this is a basic approximation and should be replaced with a proper model
        face_area=w*h
        if face_area > 10000:
            emotion_text = "Happy face detected"
            color = (0, 255, 0)
        elif face_area > 5000:
            emotion_text = "Neutral face detected"
            color = (255, 255, 0)
        else:
            emotion_text = "Sad face detected"
            color = (255, 0, 255)
        
        #put text on the videoframe
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    #display the resulting frame
    cv2.imshow('Simple Emotion Detection', frame)
    
    #break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Aala vudra gommalaka")

    
    

    
    
        