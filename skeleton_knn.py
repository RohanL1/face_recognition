import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import dlib
import os


DATA_DIR = 'img'

files = os.listdir(DATA_DIR)

image_shape = cv2.imread(os.path.join(DATA_DIR, files[0]), cv2.IMREAD_GRAYSCALE).flatten().shape

# Create the initially empty images array
images = np.empty((0, image_shape[0]+1))

print("Data dir:", DATA_DIR)
print("Loading images... ")

for file in files:
    filepath = os.path.join(DATA_DIR, file)
    # print(filepath)
    label = int(file.split('_')[1])
    
    image = cv2.imread(filepath)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp_np_array = np.array(gray_img.flatten())
    
    images = np.vstack(( images, (np.append(tmp_np_array, label)) ))
    # labels.append(file.split('_')[1])

print("Completed")

print("Dividing data into training and testing data... ")
X = images[:, :-1]  # All rows, all columns except the last one
y = images[:, -1]   # All rows, only the last column

# Get unique class labels
classes = np.unique(y)

# Split data for each class label
train_data = []
train_labels = []  # Class labels corresponding to the training data

test_data = []
test_labels = []  # Class labels corresponding to the testing data

for cls in classes:
    # Get indices of samples belonging to the current class label
    indices = np.where(y == cls)[0]
    
    # Split the indices into training and testing indices
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Use the indices to extract corresponding samples from X and y
    train_samples = X[train_indices]
    train_sample_labels = y[train_indices]  # Class labels corresponding to the training samples
    
    test_samples = X[test_indices]
    test_sample_labels = y[test_indices]  # Class labels corresponding to the testing samples
    
    # Append the training samples and labels to the respective lists
    train_data.append(train_samples)
    train_labels.append(train_sample_labels)
    
    # Append the testing samples and labels to the respective lists
    test_data.append(test_samples)
    test_labels.append(test_sample_labels)

print("Completed")
# Convert the lists to NumPy arrays
train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)

test_data = np.concatenate(test_data)
test_labels = np.concatenate(test_labels)


aligned_images_np = np.array(images)
# print(aligned_images_np)

# Create a support vector machine classifier
knn = KNeighborsClassifier(n_neighbors=len(classes))  # Create a KNN classifier with k=3

# Train the classifier
knn.fit(train_data, train_labels)

# Test the classifier
predictions = knn.predict(test_data)

# Print the accuracy of the classifier
accuracy = np.mean(predictions == test_labels)
print("Accuracy: {}%".format(accuracy * 100))
