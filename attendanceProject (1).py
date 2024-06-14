import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# here with the help of os module we are giving path of folder where we have to penentrate automatically
path = 'training_images'
# images we have in folder
images = []
# getting the images name
classNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    # reading image one by one with help of path
    currentimg = cv2.imread(f'{path}/{cl}')
    images.append(currentimg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


# step--2 finding encoding
def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dep="IT"
            f.writelines(f'\n{name},{dtString},{dep}')


encodelistknown = findEncodings(images)
print('Encoding complete')

# step-3 we are opening web cam
cap = cv2.VideoCapture(0)
# for taking images one by one from web cam
while True:
    # What cap.read() returns is a boolean (True/False) and image content. If you remove success, the img variable takes that boolean and image data as a tuple. This is why you get an error.
    success, img = cap.read()
    # size is 1/4 of real image to save time
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # now we got the image now we have to convert it to bgr
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facecurFrame = face_recognition.face_locations(imgs)
    encodecurFrame = face_recognition.face_encodings(imgs, facecurFrame)

    # her we are finding mathing faces what it will do is take face loc and grab is current face locatoion of that image simmilarly for encoding frame
    # we are using zip because we dont want them together
    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):
        # here it will compare the both the included list and our images encode list
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        print(faceDis)
        # here we will find which image is giving the least value and simply we will capture the value
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            # storing the coordinates to make rectangle
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
