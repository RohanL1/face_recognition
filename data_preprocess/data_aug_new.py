import cv2
import numpy as np
import os
import sys

input_folder = sys.argv[1]
output_folder = sys.argv[2]

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    return final_img

def adjust_contrast(image, value):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, value)
    l = np.clip(l, 0, 255)
    final_lab = cv2.merge((l, a, b))
    final_img = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    return final_img

def generate_augmented_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        output_path = os.path.join(output_folder, os.path.relpath(root, input_folder))
        os.makedirs(output_path, exist_ok=True)
        for filename in files:
            if True or filename.endswith(".jpg") or filename.endswith(".png"):
                input_path = os.path.join(root, filename)
                # output_path = os.path.join(output_path, filename.split('.')[0])

                # Load image
                h=10
                hColor=10
                templateWindowSize=7
                searchWindowSize=21
                image = cv2.imread(input_path)
                image = cv2.fastNlMeansDenoisingColored(image, None, h, hColor, templateWindowSize, searchWindowSize)


                # Generate images with different color combinations
                image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_blurred = cv2.GaussianBlur(image_grayscale, (5, 5), 0)

                # Generate images with different brightness and contrast settings
                image_bright = adjust_brightness(image, 50)
                image_dark = adjust_brightness(image, -50)
                image_high_contrast = adjust_contrast(image, 50)
                image_low_contrast = adjust_contrast(image, -50)
                
                #mirror image 
                mirrored_image = cv2.flip(image_grayscale, 1)

                # Create the output subdirectory if it doesn't exist

                # Save augmented images
                # cv2.imwrite(os.path.join(output_path, filename), image)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_grayscale.jpg"), image_grayscale)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_blurred.jpg"), image_blurred)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_bright.jpg"), image_bright)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_dark.jpg"), image_dark)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_high_contrast.jpg"), image_high_contrast)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_low_contrast.jpg"), image_low_contrast)
                cv2.imwrite(os.path.join(output_path, filename.split('.')[0] + "_mirror.jpg"), mirrored_image)

if __name__ == "__main__":
    generate_augmented_images(input_folder, output_folder)
