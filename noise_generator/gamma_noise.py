from scipy.special import gamma
import os
import cv2
import random
import shutil
import numpy as np

def add_periodic_noise(input_directory, output_directory):
    counter = 0
    for f in os.listdir(input_directory):
        image = cv2.imread(input_directory + f, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (500, 500))

        gamma_matrix = np.random.gamma(image)
        new_image = image + gamma_matrix
        max_v = np.max(new_image)
        min_v = np.min(new_image)

        for row in range(0, 500):
            for col in range(0, 500):
                new_image[row][col] = ((new_image[row][col] - min_v) / (max_v - min_v)) * 255

        cv2.imwrite("image_" + str(counter) + ".png", new_image)
        counter = counter + 1

    for f in os.listdir("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\"):
        if (f.endswith(".png")):
            shutil.move(f, output_directory)

add_periodic_noise("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\original_image\\", "C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\gamma_noise_image\\")