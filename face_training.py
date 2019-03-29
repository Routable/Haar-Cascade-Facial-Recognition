import cv2, sys
import numpy as np
from PIL import Image
import os


# Directory of our training images that we previously made.
path = 'training_data'

# Using the LBPH(LOCAL BINARY PATTERNS HISTOGRAMS) Face Recognizer that's included in OpenCV.
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Loading our Haar classifier to use for the training process.
detector = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml");


#  Returns all numbers on 5 digits to let sort the string with numeric order.
#  Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
def sort_pictures(string):
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])


# Marches through all of our images
def march_through_images(path):
    images = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for img in images:
        open_image = Image.open(img)
        img_numpy = np.array(open_image,'uint8')

        # Get's our assigned ID, as per the filename. (Used for our array)
        id = int(os.path.split(img)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids


print ("\nTraining images, please wait.")

# Driver of our training
faces, names = march_through_images(path)
recognizer.train(faces, np.array(names))

# Save our newly created model into the trainer folder inside the trainer.yml.
recognizer.write('trainer/trainer.yml')

# Print how unique faces we detected.
print("\n{0} face(s) trained!".format(len(np.unique(names))))
