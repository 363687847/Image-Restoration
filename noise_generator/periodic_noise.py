import os
import cv2
import random
import shutil
import numpy as np

def add_periodic_noise(input_directory, output_directory, noise_direction):
    counter = 0
    for f in os.listdir(input_directory):
        image = cv2.imread(input_directory + f, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (500, 500))

        rand = np.random.randint(1,100)
        frequency = np.sin(rand)
        frequency = np.abs(frequency)
        how_many_lines = frequency * rand
        how_many_lines = int(how_many_lines)
        print (how_many_lines)

        new_image = image

        if (noise_direction == "horizontal"):
            for row in range(0, 500, int(500/how_many_lines)):
                for col in range(0, 500):
                    new_image[row][col] = 255

        elif (noise_direction == "vertical"):
            for col in range(0, 500, int(500/how_many_lines)):
                for row in range(0, 500):
                    new_image[col][row] = 255

        cv2.imwrite("image_" + str(counter) + ".png", new_image)
        counter = counter + 1

    for f in os.listdir("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\"):
        if (f.endswith(".png")):
            shutil.move(f, output_directory)

add_periodic_noise("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\original_image\\", "C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\periodic_noise_image\\", "horizontal")
