from binaural import Binaural
import numpy as np
import multiprocessing as mp
import pandas as pd
from custom_timer import Timer
import data_generator_lib
from tqdm import tqdm

# Set up status bar
pbar = tqdm(total=388800)

# Configure degrees of freedom in dataset synthesis
source_azimuth_list = np.arange(0, 360, 5)
RT60_list =  [0.3, 0.5, 0.7]
SNR_list = [-20]
source_distance_from_room_centre_list =  [0.5, 1, 1.5]
mic_rotation_list =  [45, 135, 225, 315]
mic_centre_list = [np.array([1.5, 1.5]),
                   np.array([0.5, 0.5]),
                   np.array([2.5, 0.5]),
                   np.array([0.5, 2.5]),
                   np.array([2.5, 2.5])]

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
                   for source_azimuth_x in source_azimuth_list
                   for RT60 in RT60_list 
                   for SNR in SNR_list 
                   for source_distance_from_room_centre in source_distance_from_room_centre_list 
                   for mic_rotation in mic_rotation_list 
                   for mic_centre in mic_centre_list]

        output = [p.get() for p in results]

    # write csv of labels and data
    df = pd.DataFrame(output, columns=['stereo wav', 'True Azimuth'])
    df.to_csv("wav_data/data_labels.csv")
