# FACE RECOGNITION

This project aims to develop a face recognition system that can recognize and classify faces of 23 individuals. The code utilizes the Keras VGG Face model for feature extraction and Support Vector Machines (SVM) as the classifier. Additionally, the project incorporates several pre-processing techniques such as face cropping using the OpenCV Haar cascade face detector, and data augmentation methods including blurring, mirroring, changing contrast, brightness, and inverting colors.

##Installation

1.Clone the repository from GitHub:
git clone https://github.com/RohanL1/face_recognition.git

2.Install the required dependencies using pip:
pip install -r requirements.txt

##Usage
1.Ensure that you have the dataset containing images of the 23 individuals for whom you want to perform face recognition.

2.Preprocess the dataset by running the data_preprocess.sh script:
./data_preprocess.sh <input_dataset_dir> <output_dataset_dir>

This script applies face cropping using the Haar cascade face detector, as well as data augmentation techniques such as blurring, mirroring, contrast adjustment, brightness adjustment, and color inversion. The preprocessed dataset will be saved in the <output_dataset_dir> directory.

3. Train and test the face recognition model by executing the svm_face_recognition.py script:
python3 svm_face_recognition.py <train_dataset_dir> <test_dataset_dir>

This script uses the Keras VGG Face model to extract facial features from the preprocessed dataset and trains an SVM classifier. 
The trained model will be saved as trained_models/model_ser_svm_face_recognition_<TIMESTAMP>.pkl

4. Once the model is trained, you can perform face recognition on new images using the face_detect.py script:
python3 face_detect.py
The script will load the trained model and capture webcam feed to output the predicted person's name for each detected face in the webcam feed.

##Acknowledgments
Keras VGG Face: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
OpenCV: https://opencv.org/

##License
This project is licensed under the MIT License. See the LICENSE file for more details.
