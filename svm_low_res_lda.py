import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import pickle
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import train_test_split

TRAIN_DIR = 'augmented_img3'
TEST_DIR = 'test_imgs'
res = (64, 64)


def train_lda(data, labels):
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(normalized_data, labels)
    
    return lda, scaler

def transform_lda(data, lda, scaler):
    # Normalize the data using the fitted scaler
    normalized_data = scaler.transform(data)
    
    # Apply LDA transformation
    transformed_data = lda.transform(normalized_data)
    
    return transformed_data



# Load images from given DIR
def load_data_from_dir(DATA_DIR):
    global res
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
    return X,y


def get_now_string():
    return dt.datetime.now().__str__().replace(" ", "_").replace(":",'_').replace('-','_').replace('.', '_')

# Save trained model to file
def save_model_to_file(model, now_time,  prefix=''):
    model_ser_file_name = prefix + 'model_ser_' + os.path.basename(__file__).split('.')[0] + '_'+ now_time + '.pkl'
    print("Saving model in file:", model_ser_file_name)
    pickle.dump(model, open(model_ser_file_name, 'wb'))
    print("Completed.")

print("LODING DATA ...")
train_data, train_labels = load_data_from_dir(TRAIN_DIR)
test_data, test_labels = load_data_from_dir(TEST_DIR)
print("Completed")

print("LDA fit ...")
lda_model, scaler = train_lda(train_data, train_labels.flatten())
print("Completed")

print("transforming train and test data ...")
trnsfm_train_data = transform_lda(train_data, lda_model, scaler)
trnsfm_test_data = transform_lda(test_data, lda_model, scaler)
print("Completed")


# param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
# svc=svm.SVC(probability=True)
# model=GridSearchCV(svc,param_grid)
# # Accuracy: 83.96739130434783%

model = svm.SVC()
# Accuracy: 83.96739130434783%


# Train the classifier
print("Training model...")
model.fit(trnsfm_train_data, train_labels.flatten())
print("Completed")

# saving the Model for future use
now_time_string = get_now_string()
save_model_to_file(model, now_time_string)
save_model_to_file(lda_model, now_time_string, 'pca_')
save_model_to_file(scaler, now_time_string, 'scaler_')


# Test the classifier
predictions = model.predict(trnsfm_test_data)

# Print the accuracy of the classifier
accuracy = np.mean(predictions == test_labels.flatten())
print("Accuracy: {}%".format(accuracy * 100))
