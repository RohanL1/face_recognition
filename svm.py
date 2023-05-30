import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import dlib
import os
from sklearn.model_selection import GridSearchCV
import pickle
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

TRAIN_DIR = 'augmented_img3'
TEST_DIR = 'test_imgs'



def perform_lda(X, y, n_components=None):
    """
    Perform Linear Discriminant Analysis (LDA) on the given dataset.
    
    Parameters:
        X (array-like): The input data matrix of shape (n_samples, n_features).
        y (array-like): The target labels of shape (n_samples,).
        n_components (int, optional): The number of components to keep. If not provided, it keeps all components.
    
    Returns:
        array-like: The transformed data matrix of shape (n_samples, n_components).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_train, y_train.flatten())
    
    return X_lda, lda


# best pca comp number 200 
def train_pca_analysis(images, n_components=100):
    # Reshape the images to a 2D array
    num_images = len(images)
    image_shape = images[0].shape
    data = np.reshape(images, (num_images, -1))
    
    # Perform PCA analysis
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def get_trnfm_data_pca(pca, images):
    num_images = len(images)
    image_shape = images[0].shape
    # Transform the images using the learned PCA model
    transformed_data = pca.transform(images)
    
    # Reshape the transformed data back to image shape
    transformed_images = np.reshape(transformed_data, (num_images,) + image_shape)
    
    return transformed_images



def load_data_from_dir(DATA_DIR) :
    files = os.listdir(DATA_DIR)
    image_shape = cv2.imread(os.path.join(DATA_DIR, files[0])).flatten().shape
    
    X = np.empty((0, image_shape[0]))
    y = np.empty((0, 1))

    print("Loading Data from dir:", DATA_DIR)

    for file in files:
        filepath = os.path.join(DATA_DIR, file)
        # print(filepath)
        label = int(file.split('_')[1])
        
        image = cv2.imread(filepath)
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image.flatten())
        
        X = np.vstack(( X, image ))
        y = np.vstack(( y, label ))

    return X,y

print("LODING DATA ...")
train_data, train_labels = load_data_from_dir(TRAIN_DIR)
test_data, test_labels = load_data_from_dir(TEST_DIR)
print("Completed")

# print("LDA dim analysis ...")
# lda_train_data, lda_model = perform_lda(train_data, train_labels, 20)
# print("Completed")
# Create a support vector machine classifier
# model = svm.SVC(decision_function_shape='ovo')

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

# Train the classifier
print("Training model...")
model.fit(train_data, train_labels.flatten())
print("Completed")

# saving the Model 
now_time = dt.datetime.now().__str__().replace(" ", "_").replace(":",'_').replace('-','_').replace('.', '_')
model_ser_file_name = 'model_ser_' + os.path.basename(__file__) + now_time + '.pkl'
print("Saving model in file:", model_ser_file_name)
pickle.dump(model, open(model_ser_file_name, 'wb'))


# Test the classifier
# lda_test_data = lda_model.transform(test_data)
# predictions = model.predict(lda_test_data)
predictions = model.predict(test_data)

# Print the accuracy of the classifier
accuracy = np.mean(predictions == test_labels.flatten())
print("Accuracy: {}%".format(accuracy * 100))
