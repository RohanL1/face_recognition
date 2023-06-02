import cv2 
import pickle
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.svm import SVC
import numpy as np

# Load the pre-trained VGGFace model
vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

model_file = '/home/zero/ml/project/data_pre/model_ser_svm_feature_ext.py2023_05_30_03_22_04_114001.pkl'

with open(model_file, 'rb') as file:
    model = pickle.load(file)

# /home/zero/ml/project/data_pre/haarcascade_frontalface_alt2.xml
# /home/zero/ml/project/data_pre/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Extract features
def extract_features(data):
    resized_image = cv2.resize(data, (224, 224))
    preprocessed_image = preprocess_input(resized_image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    features = vggface_model.predict(preprocessed_image, verbose=0)
    return np.array(features)


def get_name_from_id(class_id):
    name_map = {
                0 : "Abdullah Khan", 
                1 : "Akshat Kalra", 
                2 : "Bowen Cheng", 
                3 : "Chakrapani Chitnis", 
                4 : "Chin Fang Hsu", 
                5 : "Edison Nalluri", 
                6 : "Kai Cong", 
                7 : "Kyle Fenole", 
                8 : "Landis Fusato", 
                9 : "Minghao Zhang", 
                10 : "Mohit Jawale", 
                11 : "Patrick Lee", 
                12 : "Peiyuan Li", 
                13 : "Rahim Firoz Chunara", 
                14 : "Rohan Vikas Lagare", 
                15 : "Sadwi Kandula", 
                16 : "Samuel Anderson", 
                17 : "Shaunak Chaudhary", 
                18 : "Tampara Venkata Santosh Anish Dora", 
                19 : "Weixuan Lin", 
                20 : "Xinyu Dong", 
                21 : "Yaoyao Peng", 
                22 : "Yufan Lin", 
}
    return name_map[class_id]


# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1: 
  
    # reads frames from a camera
    ret, img = cap.read() 
  
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        face_roi = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, (224,224))
        face_img = face_img.astype(np.float32)
        extracted_features = extract_features(face_img)
        prediction = model.predict(extracted_features)
        class_id =int(prediction[0])
        name = get_name_from_id(class_id)  # Replace with the actual name you want to display
        print(class_id, " : ", name)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
  
    # Display an image in a window
    cv2.imshow('img',img)
  
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
# Close the window
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows() 
