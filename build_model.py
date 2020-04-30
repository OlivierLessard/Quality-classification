from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.applications.densenet import DenseNet169, DenseNet121
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import NASNetMobile
from keras.optimizers import Adam


def build_finetune_model(input_shape, num_classes, show_summary):
    #base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    #base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    dropout = 0.5
    fc_layers = 512

    # defreeze all the layers
    # count_freeze = 0
    for layer in base_model.layers:
        layer.trainable = True
        # count_freeze += 1
        # if count_freeze < 30:
        #     layer.trainable = False
        # else:
        #     layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    x = Dense(fc_layers, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(fc_layers, activation='relu')(x)
    # x = Dropout(dropout)(x)
    # x = Dense(fc_layers, activation='relu')(x)

    # New soft max layer
    predictions = Dense(num_classes, activation='softmax', name='output')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    finetune_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
        finetune_model.summary()

    return finetune_model
