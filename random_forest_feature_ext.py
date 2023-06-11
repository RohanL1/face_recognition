# util pakages
import os
import datetime as dt
import numpy as np
import cv2

# pakages for image pre_processing and feature extraction
# from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.ensemble import RandomForestClassifier

# package to save model
import pickle

# train and test data dir paths
TRAIN_DIR = '/home/zero/ml/project/NEW/TRAIN/augment'
TEST_DIR = '/home/zero/ml/project/NEW/TEST/augment'
MODEL_DIR = 'trained_models'

# Load the pre-trained VGGFace model
vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# Extract features
def extract_features(data):
    print("Extractig featrures ...")
    train_features = []
    for image_data in data:
        resized_image = cv2.resize(image_data, (224, 224))
        preprocessed_image = preprocess_input(resized_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        features = vggface_model.predict(preprocessed_image, verbose=0)
        train_features.append(features.flatten())
    print("Completed.") 
    return np.array(train_features)


# Load images from given DIR
def load_data_from_dir(DATA_DIR):
    print("Loading Data from dir:", DATA_DIR)
    files = os.listdir(DATA_DIR)
    images = []
    y=[]
    for file in files:
        filepath = os.path.join(DATA_DIR, file)
        label = int(file.split('_')[1])
        image = cv2.imread(filepath)
        if image is not None:
            image = image.astype(np.float32)
            images.append(image)
            y.append(label)
        else:
            print("Failed to load image:", file)
    X = np.array(images)
    y = np.array(y) 
    print("loaded :", len(files),"files.")
    return X,y


def get_now_string():
    return dt.datetime.now().__str__().replace(" ", "_").replace(":",'_').replace('-','_').replace('.', '_')

# Save trained model to file
def save_model_to_file(model, now_time, model_dir,  prefix=''):
    model_ser_file_name = prefix + 'model_ser_' + os.path.basename(__file__).split('.')[0] + '_'+ now_time + '.pkl'
    save_path = os.path.join(model_dir, model_ser_file_name)
    print("Saving model in file:", save_path)
    pickle.dump(model, open(save_path, 'wb'))
    print("Completed.")



# Main function
if __name__ == '__main__' : 
    
    print("LODING DATA ...")
    train_data, train_labels = load_data_from_dir(TRAIN_DIR)
    test_data, test_labels = load_data_from_dir(TEST_DIR)
    print("Completed")


    # TRAIN
    # Training model : Random Forest
    print("*****TRAINING*****")
    
    # extracting features : train data 
    train_features = extract_features(train_data)
    
    print("training, please wait ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Accuracy: 95.83333333333334%
    model.fit(train_features, train_labels)

    # saving the Model for future use
    now_time_string = get_now_string()
    save_model_to_file(model, now_time_string, MODEL_DIR)

    # TEST
    # Extract features : test data
    print("*****TESTING*****")
    test_features = extract_features(test_data)

    # making predictions
    predictions = model.predict(test_features)

    # accuracy of the model
    accuracy = np.mean(predictions == test_labels.flatten())
    print("*****RESULTS*****")
    print("Accuracy: {}%".format(accuracy * 100))



