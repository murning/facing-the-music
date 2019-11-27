import numpy as np
from pyroomacoustics import ShoeBox, MicrophoneArray
import matplotlib.pyplot as plt
from celluloid import Camera


class Binaural:

    def __init__(self,
                 room_dim,
                 max_order,
                 speed_of_sound,
                 inter_aural_distance,
                 mic_height):

        self.room_dim = room_dim
        self.max_order = max_order
        self.speed_of_sound = speed_of_sound
        self.inter_aural_distance = inter_aural_distance
        self.mic_height = mic_height

    def round_azimuth(self, x, base=5):
        return base * round(x / base)

    def rt60(self, rt60):
        surface_area = sum(2 * self.room_dim[i] * self.room_dim[(i + 1) % 3]
                           for i in range(0, 3))

        volume = np.prod(self.room_dim)

        speed_of_sound = self.speed_of_sound

        absorption = ((24 * np.log(10) / speed_of_sound)
                      * (volume / (surface_area * rt60)))

        return absorption

    def noise_variance(self, SNR, distance):
        return 10 ** (-SNR / 10) / (4. * np.pi * distance) ** 2

    def source_location(self, distance, azimuth_deg):
        location = (self.room_dim / 2 + distance * np.r_[np.cos(np.deg2rad(azimuth_deg)),
                                                         np.sin(np.deg2rad(azimuth_deg)), 0.])
        return np.around(location, decimals=4)

    def mic_position(self, mic_centre, mic_rotation_deg):
        left = [mic_centre[0] - (self.inter_aural_distance / 2) * np.cos(np.deg2rad(mic_rotation_deg)),
                mic_centre[1] - (self.inter_aural_distance / 2) * np.sin(np.deg2rad(mic_rotation_deg)),
                self.mic_height]
        right = [mic_centre[0] + (self.inter_aural_distance / 2) * np.cos(np.deg2rad(mic_rotation_deg)),
                 mic_centre[1] + (self.inter_aural_distance / 2) * np.sin(np.deg2rad(mic_rotation_deg)),
                 self.mic_height]

        return np.round(np.c_[left, right], decimals=4)

    def true_azimuth(self, mic_centre, mic_rotation_degrees, source_location):

        mic_distance = self.inter_aural_distance / 2

        mic_location = np.array([mic_centre[0] + mic_distance * np.cos(np.deg2rad(mic_rotation_degrees)),
                                 mic_centre[1] + mic_distance * np.sin(np.deg2rad(mic_rotation_degrees))])

        source = np.array([source_location[0], source_location[1]])
        u = source - mic_centre
        v = mic_location - mic_centre

        theta = np.rad2deg(np.arctan2(np.linalg.det(np.array([v, u])), np.dot(v, u)))

        if theta < 0:
            theta += 360

        if round(theta) == 360.0:
            theta = 0.0
        return self.round_azimuth(theta, 5)

    def generate_impulse_pair(self, source_azimuth_degrees, source_distance_from_room_centre, SNR, RT60, mic_centre,
                              mic_rotation_degrees, fs, source_signal,
                              plot_room, plot_impulse, write_wav, wav_name):

        room = ShoeBox(self.room_dim, fs=fs, absorption=self.rt60(RT60),
                       max_order=self.max_order, sigma2_awgn=self.noise_variance(SNR, source_distance_from_room_centre))

        room.add_source(self.source_location(source_distance_from_room_centre, source_azimuth_degrees),
                        signal=source_signal)

        room.add_microphone_array(MicrophoneArray(self.mic_position(mic_centre, mic_rotation_degrees), fs=room.fs))

        room.simulate()

        true_azimuth = self.true_azimuth(mic_centre, mic_rotation_degrees,
                                         self.source_location(source_distance_from_room_centre,
                                                              source_azimuth_degrees))

        if plot_room:
            self.plot_room(mic_position=self.mic_position(mic_centre, mic_rotation_degrees), mic_centre=mic_centre,
                           source_location=self.source_location(source_distance_from_room_centre,
                                                                source_azimuth_degrees))
            print('True Azimuth: ' + str(true_azimuth))
        if plot_impulse:
            self.plot_impulse(room.mic_array.signals[1, :], room.mic_array.signals[0, :])
        if write_wav:
            room.mic_array.to_wav(wav_name + str(true_azimuth) + ".wav",
                                  norm=True,
                                  bitdepth=np.int16)

        return room.mic_array.signals[1, :], room.mic_array.signals[0, :], true_azimuth, str(true_azimuth) + ".wav"

    def plot_room(self, mic_position, mic_centre, source_location):

        plt.figure(figsize=(10, 10))

        rectangle = plt.Rectangle((0, 0), self.room_dim[0], self.room_dim[1], fill=False, lw=3)
        plt.gca().add_patch(rectangle)

        dotted_line = plt.Line2D((mic_position[0][0], mic_position[0][1]), (mic_position[1][0], mic_position[1][1]),
                                 lw=5.,
                                 marker='.',
                                 markersize=20,
                                 markerfacecolor='r',
                                 markeredgecolor='r',
                                 alpha=1)
        plt.gca().add_line(dotted_line)

        polygon = plt.Polygon(self.get_triangle_points(mic_centre, mic_position, scale_factor=4), color='g')
        plt.gca().add_patch(polygon)

        azimuth_line = plt.Line2D((mic_centre[0], source_location[0]), (mic_centre[1], source_location[1]), lw=1.,
                                  ls='-.', color='k')
        plt.gca().add_line(azimuth_line)

        circle = plt.Circle((source_location[0], source_location[1]), radius=0.1, fc='y')
        plt.gca().add_patch(circle)

        plt.axis('scaled')
        plt.title('Room Configuration', fontsize=17)
        plt.grid()


        plt.show()
        return None

    def plot_impulse(self, left, right):

        plt.figure(figsize=(10,5))
        plt.title("Real Room Impulse Response")
        plt.subplot(121)
        plt.plot(left)
        plt.title("left")
        plt.subplot(122)
        plt.title("right")
        plt.plot(right)

        plt.savefig("impulseroom.png", dpi=300)
        return None

    def get_triangle_points(self, mic_centre, mic_position, scale_factor):
        point_1 = np.ndarray.tolist(np.array([[(1 / scale_factor), 0],
                                              [0, (1 / scale_factor)]]).dot(
            np.array([mic_position[0][0] - mic_centre[0], mic_position[1][0] - mic_centre[1]]))
                                    + np.array([mic_centre[0], mic_centre[1]]))

        point_2 = np.ndarray.tolist(np.array([[(1 / scale_factor), 0],
                                              [0, (1 / scale_factor)]]).dot(
            np.array([mic_position[0][1] - mic_centre[0], mic_position[1][1] - mic_centre[1]])) + np.array(
            [mic_centre[0], mic_centre[1]]))

        point_3 = np.ndarray.tolist(np.array([[0, -1],
                                              [1, 0]]).dot(
            np.array([mic_position[0][1] - mic_centre[0], mic_position[1][1] - mic_centre[1]])).dot(
            np.array([[2, 0, ], [0, 2]])) + np.array([mic_centre[0], mic_centre[1]]))
        points = [point_1, point_2, point_3]
        return points


