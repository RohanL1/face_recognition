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
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, value):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, value)
    l = np.clip(l, 0, 255)
    final_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

def generate_augmented_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if True: #filename.endswith(".jpg") or filename.endswith(".png") or 
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.split('.')[0])

            # Load image
            print(filename)
            image = cv2.imread(input_path)
            # Generate images with different color combinations
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_inverted = cv2.bitwise_not(image)
            image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Generate images with different brightness and contrast settings
            image_bright = adjust_brightness(image, 50)
            image_dark = adjust_brightness(image, -50)
            image_high_contrast = adjust_contrast(image, 50)
            image_low_contrast = adjust_contrast(image, -50)
            
            #mirror image 
            mirrored_image = cv2.flip(image, 1)
            
            # # Generate images with different color combinations
            # mirror_image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # mirror_image_inverted = cv2.bitwise_not(image)
            # mirror_image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # # Generate images with different brightness and contrast settings
            # mirror_image_bright = adjust_brightness(image, 50)
            # mirror_image_dark = adjust_brightness(image, -50)
            # mirror_image_high_contrast = adjust_contrast(image, 50)
            # mirror_image_low_contrast = adjust_contrast(image, -50)

            # Save augmented images
            cv2.imwrite(output_path + "_original.jpg", image)
            cv2.imwrite(output_path + "_grayscale.jpg", image_grayscale)
            cv2.imwrite(output_path + "_inverted.jpg", image_inverted)
            cv2.imwrite(output_path + "_blurred.jpg", image_blurred)
            cv2.imwrite(output_path + "_bright.jpg", image_bright)
            cv2.imwrite(output_path + "_dark.jpg", image_dark)
            cv2.imwrite(output_path + "_high_contrast.jpg", image_high_contrast)
            cv2.imwrite(output_path + "_low_contrast.jpg", image_low_contrast)
            
            cv2.imwrite(output_path + "_mirror.jpg", mirrored_image)
            # cv2.imwrite(output_path + "_mirror_" + "_grayscale.jpg", mirror_image_grayscale)
            # cv2.imwrite(output_path + "_mirror_" +  "_inverted.jpg", mirror_image_inverted)
            # cv2.imwrite(output_path + "_mirror_" +  "_blurred.jpg", mirror_image_blurred)
            # cv2.imwrite(output_path + "_mirror_" +  "_bright.jpg", mirror_image_bright)
            # cv2.imwrite(output_path + "_mirror_" +  "_dark.jpg", mirror_image_dark)
            # cv2.imwrite(output_path + "_mirror_" +  "_high_contrast.jpg", mirror_image_high_contrast)
            # cv2.imwrite(output_path + "_mirror_" +  "_low_contrast.jpg", mirror_image_low_contrast)

if __name__ == "__main__":
    generate_augmented_images(input_folder, output_folder)

