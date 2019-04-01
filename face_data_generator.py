import cv2, os, time, sys, csv

# File we store our previously trained individuals.
trained_faces = os.path.isfile('trainer/trained_individuals.csv')

# For cute text animations.
def delay_print(s):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.04)


camera = cv2.VideoCapture(0)
camera.set(3, 900) # set video width
camera.set(4, 1080) # set video height
face_name_filename = 0

facial_classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')

# For each person, enter their name.
delay_print("\nEnter your name and press ENTER when ready: ")

# Take Input
face_name = input().lower()


# If our user file exists
if trained_faces:

    with open('trainer/trained_individuals.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # ID of user/name.
        exists = False
        id = 0
        total = 0

        for row in csv_reader:
            total += 1
            # Set key (persons name) to their ID
            if row:
                if row[1] == face_name:
                    exists = True
                    id = row[0]
                    face_name_filename = id

        if not exists:
                delay_print("\nYou are an unrecognized user. Adding you to the user database.")
                with open('trainer/trained_individuals.csv', mode='a') as filewriter:
                    filewriter = csv.writer(filewriter, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    new_id = total
                    filewriter.writerow([new_id, face_name])
                    face_name_filename = new_id


delay_print(("\nPlease enter how many images should be used in our dataset: "))
training_size = input()
delay_print(("\nThanks, ", face_name, ". Please look at the camera and wait."))

# Initialize individual sampling face count
count = 0

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facial_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (124, 252, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("training_data/person." + str(face_name_filename) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

        # Escape to quit.
        k = cv2.waitKey(100) & 0xff

    if count >= int(training_size):
        break


# Cleanup
delay_print("\nTraining images added!")
camera.release()
cv2.destroyAllWindows()


