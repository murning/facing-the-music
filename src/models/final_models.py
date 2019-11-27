import deep_learning_models
import constants
import utility_methods
import gccphat
import numpy as np


def gcc_cnn_init():
    """
    load model and weights of gcc-phat based CNN
    :return: model
    """
    model = deep_learning_models.m11_gcc(num_classes=constants.num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model: gcc_cnn")
    print(model.summary())
    model.load_weights("models/gcc_phat.h5")

    return model


def raw_cnn_init():
    """
        load model and weights of raw audio based CNN
        :return: model
        """
    model = deep_learning_models.m11_transfer_raw_audio(num_classes=constants.num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model: raw_cnn")
    print(model.summary())
    model.load_weights("models/raw_cnn_report.h5")

    return model


def raw_resnet_init():
    """
        load model and weights of raw audio based resnet
        :return: model
        """
    model = deep_learning_models.resnet_34_transfer(num_classes=constants.num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model: raw_resnet")
    print(model.summary())
    model.load_weights("models/resnet_report.h5")

    return model


class raw_cnn:

    def __init__(self):
        self.model = raw_cnn_init()

    def predict(self, vector):
        category = self.model.predict_classes(vector)
        doa_list = utility_methods.get_doas_from_category(category[0])
        return doa_list


class raw_resnet:

    def __init__(self):
        self.model = raw_resnet_init()

    def predict(self, vector):
        category = np.argmax(self.model.predict(vector))
        doa_list = utility_methods.get_doas_from_category(category)
        return doa_list


class gcc_cnn:

    def __init__(self):
        self.model = gcc_cnn_init()

    def predict(self, vector):
        category = self.model.predict_classes(vector)
        doa_list = utility_methods.get_doas_from_category(category[0])
        return doa_list


class gcc_dsp:

    def __init__(self):
        self.fs = constants.fs
        print("Model: gcc_dsp")

    def predict(self, vector):
        """
            Make DOA prediction based on Time Delay of Arrival Technique
            :param signal: Left Channel of Audio
            :param reference_signal: Right Channel of Audio
            :return: Two possible DOA predictions
            """

        signal = vector[0]
        reference_signal = vector[1]

        predicted_doa = 5 * round(gccphat.tdoa(signal=signal, reference_signal=reference_signal, fs=self.fs) / 5)

        possible_doas = utility_methods.get_possible_doas(predicted_doa)
        return possible_doas
