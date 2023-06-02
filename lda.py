import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import cv2

TRAIN_DIR = 'augmented_img'

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

    print("Completed")
    return X,y

# Load and flatten the image data
image_data, labels = load_data_from_dir(TRAIN_DIR)
flattened_data = image_data.reshape(image_data.shape[0], -1)

# Flatten the image data
flattened_data = image_data.reshape(image_data.shape[0], -1)

# Perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(flattened_data, labels)

# Calculate explained variance ratio
explained_variance_ratio = lda.explained_variance_ratio_

# Plot explained variance ratio
plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()
