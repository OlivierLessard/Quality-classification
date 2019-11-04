from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import balanced_accuracy_score
from __future__ import print_function
from build_model import build_finetune_model
from save_model import save_model, save_test_and_pred
from load_data import create_training_data
from metrics import test_model, print_metrics
from keras.callbacks import TensorBoard
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from keras.models import load_model
import numpy as np
import os
import keras
import pickle
import time
import sklearn


def train_model(model, x_train, y_train, x_test, y_test, x_cv, y_cv, model_name):
    # training parameters
    batch_size = 32
    epochs = 2
    data_augmentation = True

    # Prepare ModelCheckpoint callback
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode=max, verbose=1, save_best_only=True)

    NAME = "denseNet-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))    # use tensorboard --logdir=logs/ on terminal

    lr_reducer = ReduceLROnPlateau(factor=0.5, monitor='val_accuracy', mode=max, cooldown=0, patience=5, min_lr=0.5e-6)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto',
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
            rotation_range=5,
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


# parameters
num_classes = 2
img_size = 80
categories = ["Dog", "Cat"]

# load saved data
x = []
y = []
try:
    with open('x.pickle', 'rb') as data:
        x = pickle.load(data)
        data.close()
    with open('y.pickle', 'rb') as data:
        y = pickle.load(data)
        data.close()
except Exception as e:
    x, y = create_training_data(img_size, categories)
input_shape = x.shape[1:]


kf = StratifiedKFold(n_splits=5, random_state=42)
# accuracy = []
# precision = []
# recall = []
# f1 = []
# auc = []
oos_y = []
oos_pred = []
folds = 0
index_best_model = 1
best_acc = 0
x = np.reshape(x, (x.shape[0], -1))                                     # x shape: (n_samples, n_features)
for train_index, test_index in kf.split(x, y):
    folds += 1
    print(f"\n fold #{folds}")
    if folds == 1:
        x = np.reshape(x, (-1, img_size, img_size, 3))                      # x shape: (n_samples, img_size, img_size, 3)
        y = keras.utils.to_categorical(y, num_classes)

    # create cv set
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    # equalize training set
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    y_train = np.argmax(y_train, axis=1)                               # n_features
    x_train_res, y_train_res = ros.fit_resample(x_train, y_train)      # resample the train set
    x_train_res = np.reshape(x_train_res, (-1, img_size, img_size, 3))
    y_train_res = keras.utils.to_categorical(y_train_res, num_classes)

    # build the model
    if folds == 1:
        model = build_finetune_model(input_shape, num_classes, show_summary=True)
    else:
        model = build_finetune_model(input_shape, num_classes, show_summary=False)

    # train with all the specifications
    model_name = 'weights (%d).best.hdf5' % folds
    history, model = train_model(model, x_train_res, y_train_res, x_test, y_test, x_cv, y_cv, model_name)

    # prepare model evaluation
    path = r"C:\Users\olivi\PycharmProjects\DL_Project1\saved_models"
    path_model = os.path.join(path, model_name)
    model_best_weights = load_model(path_model)
    y_prediction = model_best_weights.predict(x_test, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                                 workers=1, use_multiprocessing=False)
    oos_y.append(y_test)    # [n features, n classes]
    oos_pred.append(y_prediction)

    # print fold result
    headline = f"fold #{folds}"
    print_metrics(headline, np.argmax(y_test, axis=1), np.argmax(y_prediction, axis=1))
    fold_acc = balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_prediction, axis=1))
    if fold_acc > best_acc:  # best acc according the best weights
        index_best_model = folds
        best_acc = fold_acc

    # save model (last weights)
    # model_name = f"model #{folds}"
    # save_model(model, model_name)

# build the oos y and pred
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
save_test_and_pred(oos_y, oos_pred)

# plot metrics and save cm
test_model("/n Final results : ", np.argmax(oos_y, axis=1), np.argmax(oos_pred, axis=1), categories)

