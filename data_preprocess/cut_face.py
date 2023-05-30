import cv2
import os

# Create a face detector using OpenCV's pre-trained model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Path to the directory containing the input images
# main_directory = '/home/zero/ml/project/Photos/'
main_directory = '/home/zero/ml/project/testing_data/Testing_Photos'

# Path to the directory where the extracted faces will be saved
# output_directory = '/home/zero/ml/project/face2/'
output_directory = '/home/zero/ml/project/testing_data/face/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for curr_dir in os.listdir(main_directory) : 
    # Iterate over all files in the input directory
    input_directory = os.path.join(main_directory, curr_dir)
    curr_out_dir = os.path.join(output_directory, curr_dir)
    os.makedirs(curr_out_dir, exist_ok=True)
    print("input dir : ", input_directory)
    for filename in os.listdir(input_directory):
        # Load the image
        print("filename : ", filename)
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterate over all detected faces
        cnt=0
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = image[y:y+h, x:x+w]

            resized_cropped_face = cv2.resize(face_roi, (256,256))
            
            # Generate the output filename
            output_filename = os.path.splitext(filename)[0] + '_face' + str(cnt) +  os.path.splitext(filename)[1]
            output_path = os.path.join(curr_out_dir, output_filename)
            
            # Save the face ROI as a separate image
            cv2.imwrite(output_path, resized_cropped_face)
            cnt+=1
            print(f"Face saved: {output_path}")

print("Face extraction completed.")
