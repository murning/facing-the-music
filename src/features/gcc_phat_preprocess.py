import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
from custom_timer import Timer
from gccphat import gcc_phat
import data_cnn_format as cnn
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

pbar = tqdm()

df = pd.read_csv("wav_data/data_labels.csv", usecols=[1, 2])
df = df.sample(frac=1).reset_index(drop=True)


def progress_bar(x):
    pbar.update()


pbar.total = int(len(df.index) * .9)

# Split data into train and test

df_train = df[:int(len(df.index) * .9)]
df_test = df[-int(len(df.index) * .1):]


# Functions for getting numpy arrays for train and test sets

def get_y(i, data_selection):
    l = []

    if data_selection == "train":
        l.append(df_train.iloc[i][1])
    elif data_selection == "test":
        l.append(df_test.iloc[i][1])

    else:
        print("error - train or test not selected")
        return 0

    return np.asarray(l)


def get_x_y(i, fs, data_selection, resample):
    """
    returns tuple of x, y train data pair at sample rate fs
    """

    max_tau = 0.2 / 343

    if data_selection == "train":
        file_path = "wav_data/{filename}".format(filename=df_train.iloc[i][0])
    elif data_selection == "test":
        file_path = "wav_data/{filename}".format(filename=df_test.iloc[i][0])

    else:
        print("error - train or test not selected")
        return 0

    y, sr = librosa.load(file_path, mono=False)

    if resample:
        y_8k = librosa.resample(y, sr, fs)
        x = librosa.util.fix_length(y_8k, fs)

    else:

        y_8k = y
        x = librosa.util.fix_length(y_8k, sr)

    sig = x[0]
    refsig = x[1]

    _, result_x = gcc_phat(sig, refsig, fs, max_tau)

    result_y = get_y(i, data_selection)

    return result_x, result_y


def split_x_y(output):
    x, y = zip(*output)

    X = np.asarray(x)

    Y = np.concatenate(np.asarray(y))

    return X, Y


def get_train(size, num_procs, resample, fs):
    pool = mp.Pool(processes=num_procs)

    results = [pool.apply_async(get_x_y, args=(i, fs, "train", resample), callback=progress_bar) for i in range(size)]

    output = [p.get() for p in results]

    x_train, y_train = split_x_y(output)

    return x_train, y_train


def get_test(size, num_procs, resample, fs):
    pool = mp.Pool(processes=num_procs)

    results = [pool.apply_async(get_x_y, args=(i, fs, "test", resample), callback=progress_bar) for i in range(size)]

    output = [p.get() for p in results]

    x_test, y_test = split_x_y(output)

    return x_test, y_test


if __name__ == '__main__':
    file_dir = "gcc_phat_data"


    with Timer("Training data"):
        x_train_raw, y_train_raw = get_train(int(len(df.index) * .9), 12, True, 8000)


    x_train = cnn.reshape_x_for_cnn(cnn.normalize_x_data(x_train_raw))
    np.save("{dir}/x_train.npy".format(dir=file_dir), x_train)

    y_train = to_categorical(cnn.y_to_class_id(y_train_raw))
    np.save("{dir}/y_train.npy".format(dir=file_dir), y_train)


    with Timer("Testing data"):
        x_test_raw, y_test_raw = get_test(int(len(df.index) * .1), 12, True, 8000)


    x_test = cnn.reshape_x_for_cnn(cnn.normalize_x_data(x_test_raw))
    np.save("{dir}/x_test.npy".format(dir=file_dir), x_test)

    y_test = to_categorical(cnn.y_to_class_id(y_test_raw))
    np.save("{dir}/y_test.npy".format(dir=file_dir), y_test)


