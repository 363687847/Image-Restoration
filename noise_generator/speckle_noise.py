import os
import cv2
import random
import shutil
import numpy as np

def add_speckle_noise(input_directory, output_directory):
    counter = 0
    for f in os.listdir(input_directory):
        image = cv2.imread(input_directory + f, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (500, 500))

        gauss = np.random.randn(500,500)
        gauss = gauss.reshape(500,500)        
        new_image = image + image * gauss

        cv2.imwrite("image_" + str(counter) + ".png", new_image)
        counter = counter + 1

    for f in os.listdir("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\"):
        if (f.endswith(".png")):
            shutil.move(f, output_directory)

add_speckle_noise("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\original_image\\", "C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\speckle_noise_image\\")