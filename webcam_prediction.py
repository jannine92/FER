import cv2
import numpy as np
import os
from time import sleep
import tensorflow as tf
import sys

import fer2013_predict

''' Make predictions with trained CNN: Take a picture with your
webcam (click on SPACE) and the prediction 
is calculated in fer2013_predict.py
'''

face = None
count = 1

def get_path():
    local_directory = os.path.dirname(os.path.abspath(__file__))+ '/fer2013/' 
    
    if not os.path.isdir(local_directory):
        os.mkdir(local_directory)
        
    #path = os.path.join(local_directory, 'Images/image')
    #print("path: ", path)
    
    return local_directory
 
 

def takePictures():
    video_capture = cv2.VideoCapture(0)
    
    # cascade: breaks the problem of detecting faces into multiple stages -> each block: rough test, pass: more detailed test etc.
    face_detect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_detect2 = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
    face_detect3 = cv2.CascadeClassifier("lbpcascade_profileface.xml")
    face_detect4 = cv2.CascadeClassifier("haarcascade_profileface.xml")
    
    global face
    
    
    print("--------------- Press SPACE to take picture to predict ---------------")
    
    #count1 = 1
    while True: # condition so that it doesn't take too many pictures
        
        # read returns: return code(if we have run out of time -> when reading from file), actual video frame read (one frame each loop)
        _, frame = video_capture.read() #Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
        
        #if key == 32: # press space to save an image
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale to improve detection speed and accuracy
    
        #Run classifier on frame
        face1 = face_detect1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = face_detect2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = face_detect3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = face_detect4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face1) == 1:
            face_features = face1
        elif len(face2) == 1:
            face_features = face2
        elif len(face3) == 1:
            face_features = face3
        elif len(face4) == 1:
            face_features = face4
        else:
            face_features = ""
        
        #Cut and save face
        for (x, y, w, h) in face_features: #get coordinates and size of rectangle containing face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) #draw it on the colour image "frame", with arguments: (coordinates), (size), (RGB color), line thickness 2
            face = gray[y:y+h, x:x+w] #Cut the frame to size
            
            save_image()
            
        cv2.namedWindow("Webcam")
        cv2.imshow("Webcam", frame) #Display frame
        #sleep(0.5)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition in order to work correctly, here it is bound to key 'q'
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    
def save_image():
    key = cv2.waitKey(40)
    if key == 32: # press space to save an image
        global count 
    
        try:
            face_resize = cv2.resize(face, (48, 48))
            path = get_path()
            image_name = '%s%s' % ('Images/image',count)
            path_name = os.path.join(path,image_name)
            png_image_name = path_name + '.png'
            cv2.imwrite(png_image_name, face_resize) # save image as png image
            
            pixels = np.asarray(face_resize)
            pixels = pixels.flatten()
            csv_name = path_name + '.csv'
            
            
            np.savetxt(csv_name, pixels[None], fmt='%i', delimiter=',') # [None] makes it to a 2D array with a single line
            count += 1 #Increment image number
            
            fer2013_predict.main(image_name, True)

        except:
            pass #If error, pass file
    elif key == ord('q'):
        sys.exit("End program")




def main(argv=None):
    takePictures()

if __name__ == '__main__':
    tf.app.run()