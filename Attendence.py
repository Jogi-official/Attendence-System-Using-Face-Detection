import numpy as np
import face_recognition
import cv2
import os
from datetime import  datetime

path ='Images' #path of the folder where you kept  the images
images =[]
class_names = []
my_list = os.listdir(path)
print(my_list)

for cl in my_list:
    currimg = cv2.imread(f'{path}/{cl}') #cl is the name of the name
    images.append(currimg)
    class_names.append(os.path.splitext(cl)[0]) #give only the name not jpg
print(class_names)
def findEnconding(images):
    encode_lst =[]
    for img in images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_lst.append(encode)
    return encode_lst
def MarkAttendence(name):
    with open("Attendence.csv",'r+') as f:
        my_data_list =f.readlines()
        print(my_data_list)
        namelist =[]
        time_list=[]
        for line in  my_data_list:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')



encodeListforknownfaces = findEnconding(images)
#print(len(encodeListforknownfaces))
print("Encoding Done")
cap =cv2.VideoCapture(0)
while True:
    success ,img = cap.read()
    imgSmall = cv2.resize(img , (0 , 0 ) ,None , 0.25 , 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall ,faceCurrFrame)

    for encodeface ,faceLoc in zip(encodeCurrFrame ,faceCurrFrame ):
        matches = face_recognition.compare_faces(encodeListforknownfaces , encodeface)
        faceDis = face_recognition.face_distance(encodeListforknownfaces , encodeface)
        print(faceDis) #lowest value is the match
        matchIndex =np.argmin(faceDis)

        if matches[matchIndex]:
            name = class_names[matchIndex].upper()
            print(name)
            y1 , x2  , y2  ,x1 = faceLoc
            y1, x2, y2, x1 = y1 *4, x2* 4  , y2 *4  ,x1 *4  #because we small the size 0.25 previously
            cv2.rectangle(img ,(x1  , y1)  , (x2 , y2) ,(255 , 0 , 255) , 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img,name ,(x1+6,y2-6) ,cv2.FONT_HERSHEY_COMPLEX , 0.5 , (255 ,255 ,255) , 2)
            MarkAttendence(name)

    cv2.imshow("Webcam" , img )
    cv2.waitKey(1)






