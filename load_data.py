import os
import cv2
import random
import numpy as np
import keras
import pickle

# data information
DATADIR = r"C:\Users\olivi\Desktop\PetImages - Copie"
normalize = True  # Subtracting pixel mean and divide by std


def create_training_data(x_train, y_train, x_cv, y_cv, x_test, y_test, training_data, num_classes, IMG_SIZE, CATEGORIES):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path for dogs and cats
        class_num = CATEGORIES.index(category)  # label
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    # data with random order
    random.shuffle(training_data)

    # data split train/cross validation/test
    train_split = 0.8
    cv_split = 0.1
    x_train_nbr = int(train_split*len(training_data))
    x_cv_nbr = int(cv_split*len(training_data))
    for features, label in training_data[:x_train_nbr]:
        x_train.append(features)
        y_train.append(label)
    for features, label in training_data[x_train_nbr:x_cv_nbr+x_train_nbr]:
        x_cv.append(features)
        y_cv.append(label)
    for features, label in training_data[x_train_nbr+x_cv_nbr:]:
        x_test.append(features)
        y_test.append(label)

    # transform into nd array
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE)
    x_cv = np.array(x_cv).reshape(-1, IMG_SIZE, IMG_SIZE)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE)

    x_train = np.repeat(x_train[..., np.newaxis], 3, -1)            # same 3 channels
    x_cv = np.repeat(x_cv[..., np.newaxis], 3, -1)                  # same 3 channels
    x_test = np.repeat(x_test[..., np.newaxis], 3, -1)              # same 3 channels

    y_test = np.array(y_test).reshape(-1, 1)
    y_cv = np.array(y_cv).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    # Normalize data.
    if normalize:
        x_train = x_train.astype(float)
        x_cv = x_cv.astype(float)
        x_test = x_test.astype(float)

        x_train -= np.mean(x_train, axis=0)
        x_cv -= np.mean(x_cv, axis=0)
        x_test -= np.mean(x_test, axis=0)

        x_train /= np.std(x_train, axis=0)
        x_cv /= np.std(x_cv, axis=0)
        x_test /= np.std(x_test, axis=0)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_cv shape:', x_cv.shape)
    print('y_cv shape:', y_cv.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_cv.shape[0], 'cv samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_cv = keras.utils.to_categorical(y_cv, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # save the dataset
    pickle_out = open("x_test.pickle", "wb")
    pickle.dump(x_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

    pickle_out = open("x_cv.pickle", "wb")
    pickle.dump(x_cv, pickle_out)
    pickle_out.close()

    pickle_out = open("y_cv.pickle", "wb")
    pickle.dump(y_cv, pickle_out)
    pickle_out.close()

    pickle_out = open("x_train.pickle", "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    return x_train, y_train, x_cv, y_cv, x_test, y_test
