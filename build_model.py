from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.applications.densenet import DenseNet169, DenseNet121
from keras.optimizers import Adam


def build_finetune_model(input_shape, num_classes, show_summary):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    dropout = 0.5
    fc_layers = 1024

    # defreeze all the layers
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    x = Dense(fc_layers, activation='relu')(x)
    x = Dropout(dropout)(x)

    # New soft max layer
    predictions = Dense(num_classes, activation='softmax', name='output')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    finetune_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
        finetune_model.summary()

    return finetune_model
