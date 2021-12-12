import os
import pickle
import cv2
import numpy as np

objects = []
with (open("images_cuts", "rb")) as openfile:
    while True:
        try:
            objects_images = pickle.load(openfile)
        except EOFError:
            break

with (open("label_cuts", "rb")) as openfile:
    while True:
        try:
            objects_labels = pickle.load(openfile)
        except EOFError:
            break

cv2.imshow('demo_images', objects_images[10])
cv2.imshow('demo_labels', objects_labels[10])
cv2.waitKey(0)

num_augment_images = len(objects_images)
labels_augmentation_options = ["normal", "horizontal", "vertical", "both"]

for image in os.listdir("Images/"):
    augments_on_image = np.random.randint(4, 10)
    tries = 50

    img = cv2.imread("Images/"+image)
    label = cv2.imread("Labels_Image/"+image)
    height, width, channels = img.shape

    while tries > 0 and augments_on_image != 0:
        selected_height = np.random.randint(0, height-51)
        selected_width = np.random.randint(0, width-51)

        if np.sum(label[selected_height:selected_height+51, selected_width:selected_width+51, :]) == 0:
            augment_image_number = np.random.randint(0, num_augment_images)
            images_patch = objects_images[augment_image_number]
            labels_patch = objects_labels[augment_image_number]

            augmentation_number = np.random.randint(0, len(labels_augmentation_options))
            if labels_augmentation_options[augmentation_number] == "horizontal":
                images_patch = cv2.flip(images_patch, 1)
                labels_patch = cv2.flip(labels_patch, 1)
            elif labels_augmentation_options[augmentation_number] == "vertical":
                images_patch = cv2.flip(images_patch, 0)
                labels_patch = cv2.flip(labels_patch, 0)
            elif labels_augmentation_options[augmentation_number] == "both":
                images_patch = cv2.flip(images_patch, -1)
                labels_patch = cv2.flip(labels_patch, -1)

            img[selected_height:selected_height+51, selected_width:selected_width+51, :] = images_patch
            label[selected_height:selected_height+51, selected_width:selected_width+51, :] = labels_patch

            augments_on_image -= 1
            tries = 50
        else:
            tries -= 1
    cv2.imwrite("Augmented/Images/"+image, img)
    cv2.imwrite("Augmented/Labels/"+image, label)

