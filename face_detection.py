import numpy as np
import cv2


# The following line loads our classifier.
faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')
face_label = "Face Detected"

# Enable web camera, set width and height.
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()

    # create a grey scale representation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parameters for facial detection.
    faces = faceCascade.detectMultiScale(
        gray,              # input gray scale image

        scaleFactor=1.2,   # parameter specifying how much the image size should
                           # be reduced at each image scale, used to create scale pyramid.
        minNeighbors=5,    # specifying how many neighbors each candidate rectangle should
                           # have, to retain it. Higher number gives lower false positives.
        minSize=(10, 10)   # minimum rectangle size to be considered a face.
    )

    # marks detected faces with a rectangle.
    for (x,y,w,h) in faces:

        # Top left of square is X, Y
        cv2.rectangle(img,(x,y),(x+w,y+h),(124,252,0),1)

        # Set Font/Text for Facial Recognition
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, face_label, (x,h), font, 1, (124,252,0), 1, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect Smiles
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

    cv2.imshow('Facial Detection - Data Mining/Machine Learning',img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
