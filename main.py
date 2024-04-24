import cv2 
import numpy as np
import tensorflow as tf
import h5py
import keras



# model = keras.models.load_model("./model.keras")

emotion_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Suprise"]

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
  _,frame = cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray)

  for x,y,w,h in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) 

    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = roi_gray.reshape(-1, 48, 48, 1)

    # Emotion prediction
    # predictions = model.predict(img_pixels)

    # max_index = int(np.argmax(predictions))
    # emotion = emotion_labels[max_index]

    # Display the emotion
    # cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


  # Display the resulting frame
  cv2.imshow('frame', frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
cap.release()
cv2.destroyAllWindows()
