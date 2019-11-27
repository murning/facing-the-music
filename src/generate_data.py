from binaural import Binaural
import numpy as np
import multiprocessing as mp
import pandas as pd
from custom_timer import Timer
import data_generator_lib
from tqdm import tqdm


pbar = tqdm(total=388800)

def progress(x):
    pbar.update()


if __name__ == "__main__":
    with Timer("Data Generation"):
        # Set up constant parameters: room dim etc..
        b = Binaural(room_dim=np.r_[3., 3., 2.5],
                     max_order=17,
                     speed_of_sound=343,
                     inter_aural_distance=0.2,
                     mic_height=1)

        # set up parallel processing
        pool = mp.Pool(processes=16)

        # parallel processing of data
        results = [pool.apply_async(data_generator_lib.generate_training_data,
                                    args=(
                                        source_azimuth_x,
                                        source_distance_from_room_centre,
                                        SNR,
                                        RT60,
                                        data_point_x,
                                        mic_centre,
                                        mic_rotation,
                                        b,
                                        "wav_data"),
                                    callback=progress)
                   for data_point_x in data_generator_lib.get_data(subset_size=1)
                   for source_azimuth_x in np.arange(0, 360, 5)
                   for RT60 in [0.3, 0.5, 0.7]
                   for SNR in [-20]
                   for source_distance_from_room_centre in [0.5, 1, 1.5]
                   for mic_rotation in [45, 135, 225, 315]
                   for mic_centre in [np.array([1.5, 1.5]),
                                      np.array([0.5, 0.5]),
                                      np.array([2.5, 0.5]),
                                      np.array([0.5, 2.5]),
                                      np.array([2.5, 2.5])]]

        output = [p.get() for p in results]

    # write csv of labels and data
    df = pd.DataFrame(output, columns=['stereo wav', 'True Azimuth'])
    df.to_csv("wav_data/data_labels.csv")
