import numpy as np
import cv2


# The following line loads our classifier.
faceCascade = cv2.CascadeClassifier('classifiers/haar_banana.xml')

face_label = "BANANA"

# Enable web camera + width and height.
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()

    # create a greyscale representation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parameters for facial detection.
    bananas = faceCascade.detectMultiScale(
        gray,              #input grayscale image
        scaleFactor=1.3,   #parameter specifying how much the image size shoudl be reduced at each image scale, used to create scale pyramid.
        minNeighbors=20,    #  specifying how many neighbors each candidate rectangle should have, to retain it. Highe rnumber gives lower false positives.
        minSize=(10, 10)   #minimum rectangler size to be considered a face.
    )

    # marks bananas  with a rectangle.
    for (x,y,w,h) in bananas:

        # Top left of square is X, Y
        cv2.rectangle(img, (x, y), (x+w, y+h), (124, 252, 0), 1)

        # Set Font/Text for Banana recognition
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, face_label, (x,h), font, 1, (124,252,0), 1, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('Banana Detection - Data Mining/Machine Learning',img)
    k = cv2.waitKey(30) & 0xff

cap.release()
cv2.destroyAllWindows()
