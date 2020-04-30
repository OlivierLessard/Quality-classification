from keras.models import load_model, Model
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
import pickle


def input_process(original_image):
    # Crop used for denseNet
    min_y = 100
    max_y = 300
    min_x = 240
    max_x = 400

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    original_image = original_image[min_y:max_y, min_x:max_x]  # crop contour
    original_image = np.array(original_image).reshape(-1, delta_y, delta_x)
    original_image = np.repeat(original_image[..., np.newaxis], 3, -1)
    original_image = original_image.astype(float)
    original_image -= np.mean(original_image)
    original_image /= np.std(original_image)
    np.nan_to_num(original_image, copy=False)
    return original_image


if __name__ == "__main__":
    categories = ["Bad", "Workable"]
    data_path = r"C:\Users\aiuser\Desktop\New Dataset 2 (DC Revised)"   # dataset
    database_path = r"C:\Users\aiuser\Desktop\Database"                # representation

    # Crop used for denseNet
    min_y = 100
    max_y = 300
    min_x = 240
    max_x = 400

    # Crop used for mobileNet
    # min_y = 104
    # max_y = 104 + 224
    # min_x = 240
    # max_x = 240 + 224

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    # load classifier
    model_name = "best weights (3).best.hdf5"
    model_best_weights = load_model(model_name)
    model_best_weights.summary()

    layer_name = 'dense_6'  # before softmax
    intermediate_layer_model = Model(inputs=model_best_weights.inputs, outputs=model_best_weights.get_layer(layer_name).output)

    # for each image
    for patient in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient)

        # create the directory
        database_patient_path = os.path.join(database_path, patient)
        try:
            os.mkdir(database_patient_path)
        except OSError as error:
            print("folder already exist : ", database_patient_path)


        # create the directory
        category_path = os.path.join(patient_path, 'Workable')
        database_category_path = os.path.join(database_patient_path, 'Workable')
        try:
            os.mkdir(database_category_path)
        except OSError as error:
            print("folder already exist : ", database_category_path)

        for session in os.listdir(category_path):
            session_path = os.path.join(category_path, session)

            # create the directory
            database_session_path = os.path.join(database_category_path, session)
            try:
                os.mkdir(database_session_path)
            except OSError as error:
                print("folder already exist : ", database_session_path)

            for img in os.listdir(session_path):
                database_img_path = os.path.join(database_session_path, img)
                try:
                    img_array = cv2.imread(os.path.join(session_path, img), cv2.IMREAD_GRAYSCALE)
                    img_array = input_process(img_array)

                    # calculate the representation and save it
                    representation = intermediate_layer_model.predict(img_array)
                    #cv2.imshow('representation', representation)

                    os.chdir(database_session_path)
                    pickle_out = open("{}.pickle".format(img[:-4]), "wb")
                    pickle.dump(representation, pickle_out)
                    pickle_out.close()

                except Exception as e:
                    pass


