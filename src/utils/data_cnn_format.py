from keras.utils.np_utils import to_categorical
import numpy as np
import constants
import sys


def format_for_cnn(file_dir):

    print("x_train")
    x_train_raw = np.load("{dir}/x_train_raw_388800.npy".format(dir=file_dir))
    x_train = reshape_x_for_cnn(normalize_x_data(flatten_stereo(x_train_raw)))
    np.save("{dir}/x_train.npy".format(dir=file_dir), x_train)

    print("y_train")
    y_train_raw = np.load("{dir}/y_train_raw_388800.npy".format(dir=file_dir))
    y_train = to_categorical(y_to_class_id(y_train_raw))
    np.save("{dir}/y_train.npy".format(dir=file_dir), y_train)

    print("x_test")
    x_test_raw = np.load("{dir}/x_test_raw_388800.npy".format(dir=file_dir))
    x_test = reshape_x_for_cnn(normalize_x_data(flatten_stereo(x_test_raw)))
    np.save("{dir}/x_test.npy".format(dir=file_dir), x_test)

    print("y_test")
    y_test_raw = np.load("{dir}/y_test_raw_388800.npy".format(dir=file_dir))
    y_test = to_categorical(y_to_class_id(y_test_raw))
    np.save("{dir}/y_test.npy".format(dir=file_dir), y_test)


def flatten_stereo(x_data):
    """
    flatten stereo training data into single time series
    """
    for i in x_data:
        print(i.shape)

    return np.asarray([np.concatenate(i) for i in x_data])


def normalize_zero_mean(waveform):
    output = (waveform - np.mean(waveform)) / np.std(waveform)
    return output


def normalize_x_data(x_data):

    return np.asarray([normalize_zero_mean(i) for i in x_data])


def y_to_class_id(y_data):
    """
    Converts array of doas to array of class ids based on dict

    used to get categorical data with keras
    """
    return np.asarray([constants.class_ids[str(y)] for y in y_data])


def reshape_x_for_cnn(x_data):
    return np.asarray([np.array([x]).T for x in x_data])


if __name__ == '__main__':
    format_for_cnn("raw_numpy_data")
