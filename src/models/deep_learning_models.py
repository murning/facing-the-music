import codecs
import json
import sys

import keras.backend as K
import numpy as np
from keras import layers
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import GlobalAveragePooling1D, Input
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential

import constants

RAW_AUDIO_LENGTH = 16000

GCC_LENGTH = 149


def save_hist(path, history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] = history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)


def load_hist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n


def m11_gcc(num_classes=constants.num_classes):
    print('Using Model gcc cnn')
    m = Sequential()
    m.add(Conv1D(64,
                 input_shape=[GCC_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(128,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=2, strides=None))

    # pool size above was 4, now 2
    for i in range(3):
        m.add(Conv1D(256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=1, strides=None))

    # pool size above was 4, now 1
    for i in range(2):
        m.add(Conv1D(512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
    m.add(Dense(512, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def m11_transfer_raw_audio(num_classes=constants.num_classes):
    print('Using Model raw audio cnn')
    m = Sequential()
    m.add(Conv1D(64,
                 input_shape=[RAW_AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(128,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(3):
        m.add(Conv1D(256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
    m.add(Dense(512, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def residual_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    # up-sample from the activation maps.
    # otherwise it's a mismatch. Recommendation of the authors.

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def resnet_34_transfer(num_classes=constants.num_classes):
    inputs = Input(shape=[RAW_AUDIO_LENGTH, 1])

    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = residual_block(x, kernel_size=3, filters=48, stage=1, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(4):
        x = residual_block(x, kernel_size=3, filters=96, stage=2, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(6):
        x = residual_block(x, kernel_size=3, filters=192, stage=3, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = residual_block(x, kernel_size=3, filters=384, stage=4, block=i)

    x = GlobalAveragePooling1D()(x)

    x = Dense(512, activation='relu')(x)

    x = Dense(num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='resnet34')
    return m


if __name__ == '__main__':
    num_classes = 37

    model_choice = sys.argv[1]

    model_name = sys.argv[2]

    data_dir = sys.argv[3]

    epochs = int(sys.argv[4])

    model = None

    if model_choice == 'm11_raw':

        model = m11_transfer_raw_audio(num_classes=num_classes)

    elif model_choice == 'm32':

        model = resnet_34_transfer(num_classes=num_classes)

    elif model_choice == 'm11_gcc':
        model = m11_gcc(num_classes=num_classes)

    if model is None:
        exit('error! no model')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # load data

    x_tr = np.load("{dir}/x_train.npy".format(dir=data_dir))
    y_tr = np.load("{dir}/y_train.npy".format(dir=data_dir))
    x_te = np.load("{dir}/x_test.npy".format(dir=data_dir))
    y_te = np.load("{dir}/y_test.npy".format(dir=data_dir))

    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)

    # if the accuracy does not increase over 10 epochs, reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    # if the val_loss does not decrease over 15 epochs, end the training
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)

    # save the model with the best accuracy
    model_checkpoint = ModelCheckpoint('models/{name}.h5'.format(name=model_name), monitor='val_acc', mode='max',
                                       verbose=1,
                                       save_best_only=True)

    batch_size = 128
    history = model.fit(x=x_tr,
                        y=y_tr,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_te, y_te),
                        callbacks=[reduce_lr, early_stopping, model_checkpoint])

    save_hist('training_history/HISTORY_{model_name}'.format(model_name=model_name), history)
