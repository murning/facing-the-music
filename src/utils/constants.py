from binaural import Binaural
import numpy as np

class_ids = {
    '0.0': 0,
    '5.0': 1,
    '10.0': 2,
    '15.0': 3,
    '20.0': 4,
    '25.0': 5,
    '30.0': 6,
    '35.0': 7,
    '40.0': 8,
    '45.0': 9,
    '50.0': 10,
    '55.0': 11,
    '60.0': 12,
    '65.0': 13,
    '70.0': 14,
    '75.0': 15,
    '80.0': 16,
    '85.0': 17,
    '90.0': 18,
    '95.0': 19,
    '100.0': 20,
    '105.0': 21,
    '110.0': 22,
    '115.0': 23,
    '120.0': 24,
    '125.0': 25,
    '130.0': 26,
    '135.0': 27,
    '140.0': 28,
    '145.0': 29,
    '150.0': 30,
    '155.0': 31,
    '160.0': 32,
    '165.0': 33,
    '170.0': 34,
    '175.0': 35,
    '180.0': 36,
    '185.0': 35,
    '190.0': 34,
    '195.0': 33,
    '200.0': 32,
    '205.0': 31,
    '210.0': 30,
    '215.0': 29,
    '220.0': 28,
    '225.0': 27,
    '230.0': 26,
    '235.0': 25,
    '240.0': 24,
    '245.0': 23,
    '250.0': 22,
    '255.0': 21,
    '260.0': 20,
    '265.0': 19,
    '270.0': 18,
    '275.0': 17,
    '280.0': 16,
    '285.0': 15,
    '290.0': 14,
    '295.0': 13,
    '300.0': 12,
    '305.0': 11,
    '310.0': 10,
    '315.0': 9,
    '320.0': 8,
    '325.0': 7,
    '330.0': 6,
    '335.0': 5,
    '340.0': 4,
    '345.0': 3,
    '350.0': 2,
    '355.0': 1,
    '360.0': 0,

}

inter_aural_mic_distance = 0.2

speed_of_sound = 343

max_tau = inter_aural_mic_distance / float(speed_of_sound)

# Set up constant parameters: room dim etc..
room = Binaural(room_dim=np.r_[3., 3., 2.5],
                max_order=17,
                speed_of_sound=speed_of_sound,
                inter_aural_distance=inter_aural_mic_distance,
                mic_height=1)

room_dim=np.r_[3., 3., 2.5]

num_classes = 37

mic_height=1

fs = 8000
