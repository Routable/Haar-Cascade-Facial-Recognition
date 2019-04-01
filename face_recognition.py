import cv2
import numpy as np
import os
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Our trainer file
recognizer.read('trainer/trainer.yml')

# Our classifier
cascadePath = "classifiers/haarcascade_frontalface_default.xml"


# Sets the color of our recognized name/square outlay as per our confidence level.
def color_confidence(confidence):
    confidence_color = float(confidence) * 2.55
    rgb = [255 - confidence_color, confidence_color, 0] # starting color green
    print(rgb)
    return rgb


faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = []

# Hard coded names representing the ID's of faces we have trained.
print("Populating training faces...")
with open('trainer/trained_individuals.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        if row:
            names.append(row[1])

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 1200) # camera width
cam.set(4, 800) # camera height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize=(int(minW), int(minH)),
       )

    for(x, y, w, h) in faces:

        id, confidence = recognizer.predict(gray[y:y + h, x:x+w])
        confidence_color_number = color_confidence(100 - confidence)

        # Drawing the identifier box.
        cv2.rectangle(img, (x, y), (x + w, y + h), confidence_color_number, 2)

        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < 100:
            print(names[id])
            id = names[id]
            confidence_color_number = round(100 - confidence)
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence_color_number = round(100 - confidence)
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+10,y-10), font, 1, color_confidence(confidence_color_number), 2)
        cv2.putText(img, str(confidence), (x+115,y-10), font, 1, color_confidence(confidence_color_number), 2)

    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video


cam.release()
cv2.destroyAllWindows()
