from keras.models import load_model, Model
import os
import cv2
from scipy import signal
import numpy as np
from create_database_of_representation import input_process
import pickle


def find_similar_features(poor_quality_normalized_image, model):
    database_path = r"C:\Users\aiuser\Desktop\Database"
    best_max_corr = 0
    most_similar_features_path = ""

    # Calculate features of poor_quality_image
    os.chdir(os.getcwd())
    model_name = "best weights (3).best.hdf5"
    #model_best_weights = load_model(model_name)
    model_best_weights = model  # uncomment for a real application
    #model_best_weights.summary()
    layer_name = 'dense_6'  # before softmax
    intermediate_layer_model = Model(inputs=model_best_weights.inputs, outputs=model_best_weights.get_layer(layer_name).output)
    poor_quality_features = intermediate_layer_model.predict(poor_quality_normalized_image)

    for patient in os.listdir(database_path):
        # don't use patient 06 and 07 for testing, remove the if for a real application with new images
        if patient == 'ClarityPatient06' or patient == 'ClarityPatient07':
            pass

        database_patient_path = os.path.join(database_path, patient)
        database_category_path = os.path.join(database_patient_path, 'Workable')  # check for good ones only

        for session in os.listdir(database_category_path):
            database_session_path = os.path.join(database_category_path, session)

            for img in os.listdir(database_session_path):
                database_img_path = os.path.join(database_session_path, img)
                # load the good features
                os.chdir(database_session_path)
                with open('{}'.format(img), 'rb') as data:
                        good_features = pickle.load(data)
                        data.close()
                corr = signal.correlate(good_features, poor_quality_features, mode='full')
                max_corr = np.amax(corr)

                if max_corr > best_max_corr:
                    best_max_corr = max_corr
                    most_similar_features_path = database_img_path

    return most_similar_features_path, best_max_corr


if __name__ == "__main__":
    project_path = r"C:\Users\aiuser\PycharmProjects\Quality_Classification"
    # let's use the test images of the model to test the function
    data_path_1 = r"C:\Users\aiuser\Desktop\New Dataset 2 (DC Revised)\ClarityPatient06\Bad"  # bad images unseen by the model
    data_path_2 = r"C:\Users\aiuser\Desktop\New Dataset 2 (DC Revised)\ClarityPatient07\Bad"  # bad images unseen by the model
    data_path = [data_path_1, data_path_2]

    # load  the model once
    model_name = "best weights (3).best.hdf5"
    model_best_weights = load_model(model_name)

    # Find similar features for all the bad images of Patient 06 and 07
    for path in data_path:
        for session in os.listdir(path):
            session_path = os.path.join(path, session)
            for img in os.listdir(session_path):
                # get the input processed image
                img_path = os.path.join(session_path, img)
                image_array = cv2.imread(os.path.join(img_path), cv2.IMREAD_GRAYSCALE)
                image_array = input_process(image_array)
                # find the similar features
                recommended_path, recommendation_corr = find_similar_features(image_array,model_best_weights)

                os.chdir(project_path)
                f = open("recommendation with cross correlation.txt", "a+")
                f.write("\n")
                f.write("For : {} \n".format(img_path))
                f.write("The most similar features are: {} \n".format(recommended_path))
                f.write("With a maximum value of the cross correlation is: {} \n".format(recommendation_corr))
                f.close()

            print("recommendation for ", session, " done")
