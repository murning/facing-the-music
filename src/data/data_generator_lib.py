from pyroomacoustics import datasets
import sys
import uuid


def get_data(subset_size):
    """
    Wrapper function to get sound data from the google speech commands dataset
    :param subset_size: The number of each word in the dataset
    :return: returns a list of pyroomacoustics data objects
    """

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')  # writing meaningless logging messages to trash file

    dataset = datasets.GoogleSpeechCommands(download=True, subset=subset_size, seed=0)  # get dataset

    words_in_dataset = [word for word in dataset.classes
                        if word != '_background_noise_']  # select all words in dataset without background noise

    data = [j for sub in [dataset.filter(word=selected_word)
                          for selected_word in words_in_dataset]
            for j in sub]  # create a list of data objects

    sys.stdout = save_stdout

    return data


def generate_training_data(source_azimuth_degrees, source_distance_from_room_centre, SNR, RT60, source, mic_centre,
                           mic_rotation_degrees, binaural_object, dir):
    """
    This function generates a single point of training data, taking the degrees of freedom as inputs and writing a stereo
    wav file to disk.

    The naming convention of the wav file is to use the uuid of the process in combination with the azimuth angle of
    arrival.

    The function returns the true azimuth and the wav file name.


    :param source_azimuth_degrees: The azimuth of the sound source with respect to the centre
    :param source_distance_from_room_centre: The distance of the sound source from the centre
    :param SNR: The signal to noise ratio of the source
    :param RT60: The reverberation time of the room
    :param source: The dry source signal to be used - for eg. from the google speech dataset
    :param mic_centre center of the microphone in the room eg. np.array([2, 2])
    :param mic_rotation_degrees the azimuthal rotation of the microphone
    :return: wav file name and the true azimuth
    """

    wav_name = '{uuid}_'.format(uuid=uuid.uuid4().hex)

    wav_directory = "./{dir}/{wav_name}".format(dir=dir, wav_name=wav_name)

    _, _, true_azimuth, wav_id = binaural_object.generate_impulse_pair(source_azimuth_degrees=source_azimuth_degrees,
                                                                       source_distance_from_room_centre=source_distance_from_room_centre,
                                                                       SNR=SNR,
                                                                       RT60=RT60,
                                                                       mic_centre=mic_centre,
                                                                       mic_rotation_degrees=mic_rotation_degrees,
                                                                       fs=source.fs,
                                                                       source_signal=source.data,
                                                                       plot_room=False,
                                                                       plot_impulse=False,
                                                                       write_wav=True,
                                                                       wav_name=wav_directory)

    return wav_name + wav_id, true_azimuth


def generate_simulation_data(source_azimuth_degrees, source_distance_from_room_centre, SNR, RT60, source, mic_centre,
                           mic_rotation_degrees, binaural_object, dir):
    """
    This function generates a single point of training data, taking the degrees of freedom as inputs and writing a stereo
    wav file to disk.

    The naming convention of the wav file is to use the uuid of the process in combination with the azimuth angle of
    arrival.

    The function returns the true azimuth and the wav file name.


    :param source_azimuth_degrees: The azimuth of the sound source with respect to the centre
    :param source_distance_from_room_centre: The distance of the sound source from the centre
    :param SNR: The signal to noise ratio of the source
    :param RT60: The reverberation time of the room
    :param source: The dry source signal to be used - for eg. from the google speech dataset
    :param mic_centre center of the microphone in the room eg. np.array([2, 2])
    :param mic_rotation_degrees the azimuthal rotation of the microphone
    :return: wav file name and the true azimuth
    """

    wav_name = '{uuid}_'.format(uuid=uuid.uuid4().hex)

    wav_directory = "./{dir}/{wav_name}".format(dir=dir, wav_name=wav_name)

    _, _, true_azimuth, wav_id = binaural_object.generate_impulse_pair(source_azimuth_degrees=source_azimuth_degrees,
                                                                       source_distance_from_room_centre=source_distance_from_room_centre,
                                                                       SNR=SNR,
                                                                       RT60=RT60,
                                                                       mic_centre=mic_centre,
                                                                       mic_rotation_degrees=mic_rotation_degrees,
                                                                       fs=source.fs,
                                                                       source_signal=source.data,
                                                                       plot_room=True,
                                                                       plot_impulse=False,
                                                                       write_wav=True,
                                                                       wav_name=wav_directory)

    return wav_name + wav_id, true_azimuth
