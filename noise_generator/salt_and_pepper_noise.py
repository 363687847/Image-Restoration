import os
import cv2
import random
import shutil

def add_salt_and_pepper_noise(input_directory, output_directory):
    counter = 0
    for f in os.listdir(input_directory):
        image = cv2.imread(input_directory + f, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (500, 500))
        new_image = image

        for each in range(0, 1000):
            random_x = random.randint(0, 499)
            random_y = random.randint(0, 499)
            new_image[random_x, random_y] = 255

        for each in range(0, 1000):
            random_x = random.randint(0, 499)
            random_y = random.randint(0, 499)
            new_image[random_x, random_y] = 0

        cv2.imwrite("image_" + str(counter) + ".png", new_image)
        counter = counter + 1

    for f in os.listdir("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\"):
        if (f.endswith(".png")):
            shutil.move(f, output_directory)

add_salt_and_pepper_noise("C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\original_image\\", "C:\\Users\\marko\\Desktop\\UH\\COSC\\COSC4393\\GA\\noise_generator\\salt_and_pepper_noise_image\\")