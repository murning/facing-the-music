import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
from custom_timer import Timer

from tqdm import tqdm

pbar = tqdm()



df = pd.read_csv("wav_data/data_labels.csv", usecols=[1,2])
df = df.sample(frac=1).reset_index(drop=True)



def progress_bar(x):
    pbar.update()


pbar.total = int(len(df.index) * .9)


# Split data into train and test

df_train = df[:int(len(df.index) * .9)]  # training set of 45000
df_test = df[-int(len(df.index) * .1):]  # test set of 11160



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
        result_x = np.zeros((2, fs))
        result_x = librosa.util.fix_length(y_8k, fs)

    else:

        y_8k = y
        result_x = np.zeros((2, sr))
        result_x = librosa.util.fix_length(y_8k, sr)

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

    print("loading and processing training data \n")

    with Timer("Training data"):
        x_train_raw, y_train_raw = get_train(int(len(df.index) * .9), 12, True, 8000)


    print("saving raw training data \n")

    np.save("x_train_raw_388800.npy", x_train_raw)
    np.save("y_train_raw_388800.npy", y_train_raw)


    print("training data shape: \n")

    print("x_train: {data} \n".format(data=x_train_raw.shape))

    print("y_train: {data} \n".format(data=y_train_raw.shape))

    print("----------------------------------------------------------")


    print("loading and processing testing data \n")

    with Timer("Testing data"):
        x_test_raw, y_test_raw = get_test(int(len(df.index) * .1), 12, True, 8000)



    print("saving raw testing data \n")
    np.save("x_test_raw_388800.npy", x_test_raw)
    np.save("y_test_raw_388800.npy", y_test_raw)

    print("testing data shape: \n")

    print("x_test: {data} \n".format(data=x_test_raw.shape))

    print("y_test: {data} \n".format(data=y_test_raw.shape))



