import cv2
import numpy as np
from sklearn.svm import SVC
import os
# from sklearn.model_selection import GridSearchCV
import pickle
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

import sys


# train and test data dir paths
# TRAIN_DIR = '/home/zero/ml/project/NEW/TRAIN/augment'
# TEST_DIR = '/home/zero/ml/project/NEW/TEST/combined'
MODEL_DIR = 'trained_models'

res = (64,64) # image res


def train_pca(data, n_components):
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    # Train PCA
    pca = PCA(n_components=n_components)
    pca.fit(normalized_data)
    
    return pca, scaler

def transform_pca(data, pca, scaler):
    # Normalize the data using the fitted scaler
    normalized_data = scaler.transform(data)
    # Apply PCA transformation
    transformed_data = pca.transform(normalized_data)
    
    return transformed_data


# Load images from given DIR
def load_data_from_dir_res(DATA_DIR, res):
    print("Loading Data from dir:", DATA_DIR)
    files = os.listdir(DATA_DIR)
    images = []
    y=[]
    for file in files:
        filepath = os.path.join(DATA_DIR, file)
        label = int(file.split('_')[1])
        image = cv2.imread(filepath)
        image = cv2.resize(image, res)
        if image is not None:
            image = image.astype(np.float32)
            images.append(image.flatten())
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
    
    TRAIN_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
    
    print("LODING DATA ...")
    train_data, train_labels = load_data_from_dir_res(TRAIN_DIR, res)
    test_data, test_labels = load_data_from_dir_res(TEST_DIR, res)
    print("Completed")

    print("PCA fit ...")
    pca_model, scaler = train_pca(train_data, n_components=200)
    print("Completed")

    print("transforming train and test data ...")
    trnsfm_train_data = transform_pca(train_data, pca_model, scaler)
    trnsfm_test_data = transform_pca(test_data, pca_model, scaler)
    print("Completed")


    # param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    # svc=SVC()
    # model=GridSearchCV(svc,param_grid)

    model = SVC()
    # Accuracy: 71.38888888888889%

    # Train the classifier
    print("Training model...")
    model.fit(trnsfm_train_data, train_labels.flatten())
    print("Completed")

    # saving the Model for future use
    now_time_string = get_now_string()
    save_model_to_file(model, now_time_string, MODEL_DIR)
    save_model_to_file(pca_model, now_time_string, MODEL_DIR, 'pca_')
    save_model_to_file(scaler, now_time_string, MODEL_DIR, 'scaler_')


    # Test the classifier
    predictions = model.predict(trnsfm_test_data)

    # Print the accuracy of the classifier
    accuracy = np.mean(predictions == test_labels.flatten())
    print("Accuracy: {}%".format(accuracy * 100))
