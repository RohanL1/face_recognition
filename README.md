# FACE RECOGNITION

This project aims to develop a face recognition system that can recognize and classify faces of 23 individuals. The code utilizes the Keras VGG Face model for feature extraction and Support Vector Machines (SVM) as the classifier. Additionally, the project incorporates several pre-processing techniques such as face cropping using the OpenCV Haar cascade face detector, and data augmentation methods including blurring, mirroring, changing contrast, brightness, and inverting colors.</br>

**Installation**</br>

1.Clone the repository from GitHub:</br>
`git clone https://github.com/RohanL1/face_recognition.git`</br>

2.Install the required dependencies using pip:</br>
`pip install -r requirements.txt`

NOTE:
if you get ModuleNotFoundError: No module named 'keras.engine.topology', do the following to resolve it.
change the import from from keras.engine.topology import get_source_inputs
to
from keras.utils.layer_utils import get_source_inputs in keras_vggface/models.py

**Usage**</br>
1.Ensure that you have the dataset containing images of the 23 individuals for whom you want to perform face recognition.</br>

2.Preprocess the dataset by running the data_preprocess.sh script:</br>
`./data_preprocess.sh <input_dataset_dir> <output_dataset_dir>`
</br>
This script applies face cropping using the Haar cascade face detector, as well as data augmentation techniques such as blurring, mirroring, contrast adjustment, brightness adjustment, and color inversion. The preprocessed dataset will be saved in the <output_dataset_dir> directory.</br>

3. Train and test the face recognition model by executing the svm_face_recognition.py script:</br>
`python3 svm_face_recognition.py <train_dataset_dir> <test_dataset_dir>`
</br>
This script uses the Keras VGG Face model to extract facial features from the preprocessed dataset and trains an SVM classifier. </br>
The trained model will be saved as trained_models/model_ser_svm_face_recognition_<TIMESTAMP>.pkl</br>


5. Once the model is trained, you can perform live face recognition on new images using the face_detect.py script:</br>
`python3 face_detect.py <MODEL_PATH>`
 </br>
The script will load the trained model and capture webcam feed to output the predicted person's name for each detected face in the webcam feed.

NOTE: update model path accordingly 

</br>
**Acknowledgments**</br>
Keras VGG Face: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/</br>
OpenCV: https://opencv.org/</br>
</br>
**License**</br>
This project is licensed under the MIT License. See the LICENSE file for more details.</br>
