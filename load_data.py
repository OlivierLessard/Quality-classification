import os
import cv2
import random
import numpy as np
import keras
import pickle
from matplotlib import pyplot as plt


def create_training_data(IMG_SIZE, categories):
    # data information
    #data_path = r"C:\Users\olivi\Desktop\PetImages - Copie"
    data_path = r"D:\Dataset2_jpg"
    training_data = []

    for category in categories:
        path = os.path.join(data_path, category)
        class_num = categories.index(category)  # label
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #img_array = img_array[47:447, 78:559]   # crop contour
                img_array = img_array[100:300, 240:400]   # zoom
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    # data with random order
    random.shuffle(training_data)

    x = []
    y = []

    for features, label in training_data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE)     # nd array
    x = np.repeat(x[..., np.newaxis], 3, -1)            # same 3 channels

    y = np.array(y).reshape(-1, 1)

    # normalize
    x = x.astype(float)
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    np.nan_to_num(x, copy=False)

    print('x shape:', x.shape)
    print('y shape:', y.shape)

    # save the dataset
    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    return x, y

if __name__ == "__main__":
    categories = ["Bad", "Workable"]
    create_training_data(80, categories)

