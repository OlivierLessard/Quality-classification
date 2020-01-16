from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import balanced_accuracy_score
from build_model import build_finetune_model
from save_model import save_model, save_test_and_pred, save_y_test_and_pred
from load_data import create_training_data
from metrics import test_model, print_metrics
from keras.callbacks import TensorBoard
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model
import numpy as np
import os
import keras
import pickle
import time


def hard_stratified_kfold_patient_split():
    # split 5 times the ordered data. Ordered by category, session, im
    test_index_1 = np.arange(145)       # clarityPatient Bad
    test_index_2 = np.arange(145, 206)  # David Bad
    test_index_3 = np.arange(221, 235)  # heath bad
    test_index_4 = np.arange(206, 221)  # george bad

    test_index_1 = np.append(test_index_1, np.arange(285, 415))  # clarityPatient workable
    test_index_2 = np.append(test_index_2, np.arange(415, 478))  # david workable
    test_index_2 = np.append(test_index_2, np.arange(531, 568))  # j-c workable
    test_index_3 = np.append(test_index_3, np.arange(246, 274))  # louis bad
    test_index_3 = np.append(test_index_3, np.arange(658, 709))  # Ron bad
    test_index_4 = np.append(test_index_4, np.arange(235, 246))  # jeff bad
    test_index_4 = np.append(test_index_4, np.arange(274, 285))  # pierre bad
    test_index_4 = np.append(test_index_4, np.arange(478, 510))  # george workable
    test_index_4 = np.append(test_index_4, np.arange(568, 603))  # jeff workable
    test_index_4 = np.append(test_index_4, np.arange(635, 658))  # pierre workable

    val_index_1 = test_index_2
    val_index_2 = test_index_3
    val_index_3 = test_index_4
    val_index_4 = test_index_1

    all_index = np.arange(709)
    train_index_1 = np.delete(all_index, np.append(test_index_1, val_index_1))
    train_index_2 = np.delete(all_index, np.append(test_index_2, val_index_2))
    train_index_3 = np.delete(all_index, np.append(test_index_3, val_index_3))
    train_index_4 = np.delete(all_index, np.append(test_index_4, val_index_4))

    return [train_index_1, val_index_1, test_index_1], [train_index_2, val_index_2, test_index_2], [train_index_3, val_index_3, test_index_3], [train_index_4, val_index_4, test_index_4]

def hard_stratified_kfold_session_split():
    # split 5 times the ordered data. Ordered by category, session, im
    test_index_1 = np.arange(57)        # correspond to 3 different sessions (Bad)
    test_index_2 = np.arange(57, 114)   # 3 other sessions (same patient)
    test_index_3 = np.arange(114, 176)
    test_index_4 = np.arange(176, 235)
    test_index_5 = np.arange(235, 285)

    test_index_1 = np.append(test_index_1, np.arange(285, 349))    # add good sessions
    test_index_2 = np.append(test_index_2, np.arange(349, 400))
    test_index_3 = np.append(test_index_3, np.arange(400, 415))
    test_index_2 = np.append(test_index_2, np.arange(415, 436))    # to balance
    test_index_3 = np.append(test_index_3, np.arange(436, 510))
    test_index_4 = np.append(test_index_4, np.arange(510, 603))
    test_index_1 = np.append(test_index_1, np.arange(603, 620))    # to balance
    test_index_5 = np.append(test_index_5, np.arange(620, 709))

    val_index_1 = test_index_2
    val_index_2 = test_index_3
    val_index_3 = test_index_4
    val_index_4 = test_index_5
    val_index_5 = test_index_1

    all_index = np.arange(709)
    train_index_1 = np.delete(all_index, np.append(test_index_1, val_index_1))  # delete val_index
    train_index_2 = np.delete(all_index, np.append(test_index_2, val_index_2))
    train_index_3 = np.delete(all_index, np.append(test_index_3, val_index_3))
    train_index_4 = np.delete(all_index, np.append(test_index_4, val_index_4))
    train_index_5 = np.delete(all_index, np.append(test_index_5, val_index_5))

    return [train_index_1, val_index_1, test_index_1], [train_index_2, val_index_2, test_index_2], [train_index_3, val_index_3, test_index_3], [train_index_4, val_index_4, test_index_4], [train_index_5, val_index_5, test_index_5]


def train_model(model, x_train, y_train, x_cv, y_cv, model_name):
    # training parameters
    batch_size = 16
    epochs = 100
    data_augmentation = True

    # Prepare callbacks
    checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_accuracy', mode=max, verbose=1, save_best_only=True)

    NAME = "denseNet-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))    # use tensorboard --logdir=logs/ on terminal

    lr_reducer = ReduceLROnPlateau(factor=0.5, monitor='val_accuracy', mode=max, cooldown=0, patience=5, min_lr=0.5e-6)

    earlystopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=False)
    callbacks = [checkpoint, lr_reducer, tensorboard, earlystopping]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_cv, y_cv),
                              shuffle=True,
                              callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=20,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=False,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_cv, y_cv),
                                      epochs=epochs, verbose=1, workers=4, callbacks=callbacks)

    return history, model


if __name__ == "__main__":
    # parameters
    num_classes = 2
    categories = ["Bad", "Workable"]

    # Get the data
    x, y = create_training_data(categories)
    input_shape = x.shape[1:]

    # hard stratified K fold
    oos_y, oos_pred = [], []
    folds, index_best_model, best_acc = 0, 1, 0
    x = np.reshape(x, (x.shape[0], -1))  # (n_samples, n_features)

    for train_index, val_index, test_index in hard_stratified_kfold_patient_split():
        folds += 1
        print("\n fold #{}".format(folds))
        #if folds == 1:
        if folds < 4:
            with open("y_pred_{}.pickle".format(folds), 'rb') as data:
                y_prediction = pickle.load(data)
                data.close()
            with open("y_test_{}.pickle".format(folds), 'rb') as data:
                y_test = pickle.load(data)
                data.close()
            oos_y.append(y_test)  # [n features, n classes]
            oos_pred.append(y_prediction)  # [n features, n classes]
            continue
        if folds == 4:
            x = np.reshape(x, (-1, input_shape[0], input_shape[1], 3))  # (n_samples, delta_y, delta_x, 3)
            y = keras.utils.to_categorical(y, num_classes)  # (n_samples, nb_classes)

        # create sets, index are shuffled
        x_train, x_cv, x_test = x[train_index], x[val_index], x[test_index]
        y_train, y_cv, y_test = y[train_index], y[val_index], y[test_index]

        # equalize training set and validation set
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        y_train = np.argmax(y_train, axis=1)                               # n_features
        x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)      # resample the bad ones
        x_train_ros = np.reshape(x_train_ros, (-1, input_shape[0], input_shape[1], 3))
        y_train_ros = keras.utils.to_categorical(y_train_ros, num_classes)

        x_cv = np.reshape(x_cv, (x_cv.shape[0], -1))
        y_cv = np.argmax(y_cv, axis=1)
        x_cv_ros, y_cv_ros = ros.fit_resample(x_cv, y_cv)
        x_cv_ros = np.reshape(x_cv_ros, (-1, input_shape[0], input_shape[1], 3))
        y_cv_ros = keras.utils.to_categorical(y_cv_ros, num_classes)

        # shuffle again after oversampling
        np.random.seed(0)
        np.random.shuffle(y_train_ros)
        np.random.seed(0)
        np.random.shuffle(x_train_ros)

        # build the model
        if folds == 1:
            model = build_finetune_model(input_shape, num_classes, show_summary=True)
        else:
            model = build_finetune_model(input_shape, num_classes, show_summary=False)

        # train with all the specifications
        model_name = 'weights ({}).best.hdf5'.format(folds)
        history, model = train_model(model, x_train_ros, y_train_ros, x_cv_ros, y_cv_ros, model_name)

        # get the best model for evaluation
        model_best_weights = load_model(model_name)
        y_prediction = model_best_weights.predict(x_test, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                                                  workers=1, use_multiprocessing=False)

        save_y_test_and_pred(y_prediction, y_test, folds)
        oos_y.append(y_test)                # [n features, n classes]
        oos_pred.append(y_prediction)       # [n features, n classes]

        # print fold result
        headline = "fold #{}".format(folds)
        print_metrics(headline, np.argmax(y_test, axis=1), np.argmax(y_prediction, axis=1))
        fold_acc = balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_prediction, axis=1))
        if fold_acc > best_acc:  # best acc according the best weights
            index_best_model = folds
            best_acc = fold_acc

    # build the oos y and pred
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    save_test_and_pred(oos_y, oos_pred)

    # plot metrics and save cm
    test_model("/n Final results : ", np.argmax(oos_y, axis=1), np.argmax(oos_pred, axis=1), categories)

