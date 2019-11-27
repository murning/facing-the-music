import pandas as pd
import multiprocessing as mp
import numpy as np
import librosa
from tqdm import tqdm

pbar = tqdm()


# def wav_to_np_single(num_procs=4, resample=True, fs=8000, dir="single_data_point"):
#     df = load_data(dir)
#
#     pbar.total = len(df.index)
#
#     x_raw, y_raw = get_raw_np(size=len(df.index),
#                               num_procs=num_procs,
#                               resample=resample,
#                               fs=fs,
#                               df=df,
#                               directory=dir)
#
#     x_, y_ = dataloading.format_for_cnn(x_raw, y_raw)
#
#     return x_, y_


def wav_to_np(num_procs=4, resample=True, fs=8000, wav_dir="wav_data", file_dir="raw_numpy_data"):
    df = load_data(wav_dir)

    pbar.total = len(df.index)

    df_train, df_test = split_df_train_test(df)

    # x_train_raw, y_train_raw = get_raw_np(size=len(df_train.index),
    #                                       num_procs=num_procs,
    #                                       resample=resample,
    #                                       fs=fs,
    #                                       df=df_train,
    #                                       directory=wav_dir)
    #
    # x_test_raw, y_test_raw = get_raw_np(size=len(df_test.index),
    #                                     num_procs=num_procs,
    #                                     resample=resample,
    #                                     fs=fs,
    #                                     df = df_test,
    #                                     directory=wav_dir)

    pool = mp.Pool(processes=num_procs)

    results_train = [pool.apply_async(get_x_y, args=(i, fs, resample, wav_dir, df_train), callback=progress_bar) for i
                     in
                     range(len(df_train.index))]

    output_train = [p.get() for p in results_train]

    pool.close()
    pool.join()

    x_train_raw, y_train_raw = split_x_y(output_train)

    np.save("{dir}/x_train_raw.npy".format(dir=file_dir), x_train_raw)
    np.save("{dir}/y_train_raw.npy".format(dir=file_dir), y_train_raw)

    results_test = [pool.apply_async(get_x_y, args=(i, fs, resample, wav_dir, df_test), callback=progress_bar) for i in
                    range(len(df_test.index))]

    output_test = [p.get() for p in results_test]

    pool.close()
    pool.join()

    x_test_raw, y_test_raw = split_x_y(output_test)

    np.save("{dir}/x_test_raw.npy".format(dir=file_dir), x_test_raw)
    np.save("{dir}/y_test_raw.npy".format(dir=file_dir), y_test_raw)


def load_data(directory):
    file_path = "{directory}/data_labels.csv".format(directory=directory)

    # load data csv
    df = pd.read_csv(file_path, usecols=[1, 2])

    # Shuffle the Dataframe so there aren't inherent patterns in the data
    return df.sample(frac=1).reset_index(drop=True)


def split_df_train_test(df):
    # Split data into train and test
    training_set_size, test_set_size = get_set_sizes(df)

    df_train = df[:training_set_size]  # training set
    df_test = df[-test_set_size:]  # test set

    return df_train, df_test


def get_set_sizes(df):
    """
    get sizes of training and test sets based on size of dataset.
    testing set: 10% of training set
    :return: training_set_size, testing_set_size
    """

    data_set_size = len(df.index)

    training_set_size = data_set_size * 0.9
    testing_set_size = data_set_size * 0.1

    return int(training_set_size), int(testing_set_size)


def progress_bar(x):
    pbar.update()


def get_raw_np(size, num_procs, resample, fs, df, directory):
    pool = mp.Pool(processes=num_procs)

    results = [pool.apply_async(get_x_y, args=(i, fs, resample, directory, df), callback=progress_bar) for i in
               range(size)]

    output = [p.get() for p in results]

    pool.close()
    pool.join()

    x_, y_ = split_x_y(output)

    return x_, y_


def get_x_y(i, fs, resample, directory, input_data_frame):
    """
    returns tuple of x, y  data pair at sample rate fs
    written to ensure that parallel computations have the correct label

    """

    file_path = "{directory}/{filename}".format(directory=directory, filename=input_data_frame.iloc[i][0])

    y, sr = librosa.load(file_path, mono=False)

    if resample:
        y_8k = librosa.resample(y, sr, fs)
        result_x = librosa.util.fix_length(y_8k, fs)

    else:

        y_8k = y
        result_x = librosa.util.fix_length(y_8k, sr)

    result_y = get_y(i, input_data_frame)

    return result_x, result_y


def get_y(i, df):
    l = []

    l.append(df.iloc[i][1])

    return np.asarray(l)


def split_x_y(output):
    x, y = zip(*output)

    X = np.asarray(x)

    Y = np.concatenate(np.asarray(y))

    return X, Y


if __name__ == '__main__':
    num_procs=16
    resample=True
    fs=8000
    wav_dir="wav_data"
    file_dir="raw_numpy_data"

    df = load_data(wav_dir)


    df_train, df_test = split_df_train_test(df)

    pbar.total = len(df_train.index)



    pool = mp.Pool(processes=num_procs)

    results_train = [pool.apply_async(get_x_y, args=(i, fs, resample, wav_dir, df_train), callback=progress_bar) for i
                     in
                     range(len(df_train.index))]

    output_train = [p.get() for p in results_train]


    x_train_raw, y_train_raw = split_x_y(output_train)

    np.save("{dir}/x_train_raw.npy".format(dir=file_dir), x_train_raw)
    np.save("{dir}/y_train_raw.npy".format(dir=file_dir), y_train_raw)

    # results_test = [pool.apply_async(get_x_y, args=(i, fs, resample, wav_dir, df_test), callback=progress_bar) for i in
    #                 range(len(df_test.index))]
    #
    # output_test = [p.get() for p in results_test]
    #
    # pool.close()
    # pool.join()
    #
    # x_test_raw, y_test_raw = split_x_y(output_test)
    #
    # np.save("{dir}/x_test_raw.npy".format(dir=file_dir), x_test_raw)
    # np.save("{dir}/y_test_raw.npy".format(dir=file_dir), y_test_raw)



