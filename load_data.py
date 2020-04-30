import os
import cv2
import random
import numpy as np
import keras
import pickle
from matplotlib import pyplot as plt
import argparse

# Crop used for denseNet
min_y = 100
max_y = 300
min_x = 240
max_x = 400

# Crop used for mobileNet
# min_y = 104
# max_y = 104+224
# min_x = 240
# max_x = 240+224

delta_x = max_x - min_x
delta_y = max_y - min_y


def create_training_data(categories):
    data_path = r"C:\Users\aiuser\Desktop\New Dataset 2 (DC Revised)"
    training_data = []
    x = []
    y = []

    # load x, y
    try:
        with open('x.pickle', 'rb') as data:
            x = pickle.load(data)
            data.close()
        with open('y.pickle', 'rb') as data:
            y = pickle.load(data)
            data.close()
        return x, y
    except Exception as e:
        pass

    # create x, y
    for patient in os.listdir(data_path):                           # ClarityPatientXX
        patient_path = os.path.join(data_path, patient)
        for category in os.listdir(patient_path):                   # bad, workable
            category_path = os.path.join(patient_path, category)
            class_num = categories.index(category)                  # label
            for session in os.listdir(category_path):               # session
                session_path = os.path.join(category_path, session)
                for img in os.listdir(session_path):
                    try:
                        img_array = cv2.imread(os.path.join(session_path, img), cv2.IMREAD_GRAYSCALE)
                        # crop contour
                        img_array = img_array[min_y:max_y, min_x:max_x]

                        training_data.append([img_array, class_num])
                    except Exception as e:
                        pass

    for features, label in training_data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, delta_y, delta_x)       # nd array
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


def create_fold_data():
    from train_model import hard_stratified_kfold
    from imblearn.over_sampling import RandomOverSampler

    num_classes = 2
    categories = ["Bad", "Workable"]

    # Get the data
    x, y = create_training_data(categories)
    input_shape = x.shape[1:]

    x = np.reshape(x, (x.shape[0], -1))  # (n_samples, n_features)
    folds = 0
    for train_index, val_index, test_index in hard_stratified_kfold():
        folds += 1
        if folds == 1:
            x = np.reshape(x, (-1, delta_y, delta_x, 3))  # (n_samples, delta_y, delta_x, 3)
            y = keras.utils.to_categorical(y, num_classes)  # (n_samples, nb_classes)

        # create sets, index are shuffled
        x_train, x_cv, x_test = x[train_index], x[val_index], x[test_index]
        y_train, y_cv, y_test = y[train_index], y[val_index], y[test_index]

        # equalize training set
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        y_train = np.argmax(y_train, axis=1)                               # n_features
        x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)      # resample the bad ones
        x_train_ros = np.reshape(x_train_ros, (-1, delta_y, delta_x, 3))
        y_train_ros = keras.utils.to_categorical(y_train_ros, num_classes)

        # shuffle again after oversampling
        np.random.seed(0)
        np.random.shuffle(y_train_ros)
        np.random.seed(0)
        np.random.shuffle(x_train_ros)

        # save
        pickle_out = open("x_train_ros_{}.pickle".format(folds), "wb")
        pickle.dump(x_train_ros, pickle_out)
        pickle_out.close()

        pickle_out = open("y_train_ros_{}.pickle".format(folds), "wb")
        pickle.dump(y_train_ros, pickle_out)
        pickle_out.close()

        pickle_out = open("x_cv_{}.pickle".format(folds), "wb")
        pickle.dump(x_cv, pickle_out)
        pickle_out.close()

        pickle_out = open("y_cv_{}.pickle".format(folds), "wb")
        pickle.dump(y_cv, pickle_out)
        pickle_out.close()

        pickle_out = open("x_test_{}.pickle".format(folds), "wb")
        pickle.dump(x_test, pickle_out)
        pickle_out.close()

        pickle_out = open("y_test_{}.pickle".format(folds), "wb")
        pickle.dump(y_test, pickle_out)
        pickle_out.close()


if __name__ == "__main__":
    categories = ["Bad", "Workable"]
    create_training_data(categories)
    #create_fold_data()


