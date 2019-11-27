import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
import constants
from matplotlib.offsetbox import AnchoredText
from IPython.display import HTML


class plotting:
    def __init__(self, room_dim, source_distance, source_azimuth, mic_centre, rotation_list, prediction_list,
                 prediction):
        self.fig = plt.figure(figsize=(10, 10))
        self.camera = Camera(self.fig)
        self.room_dim = room_dim
        self.source_distance = source_distance
        self.source_azimuth = source_azimuth
        self.mic_centre = mic_centre
        self.rotation_list = rotation_list
        self.prediction_list = prediction_list
        self.prediction = prediction

    def mic_position(self, mic_centre, mic_rotation_deg):
        left = [mic_centre[0] - (constants.inter_aural_mic_distance / 2) * np.cos(np.deg2rad(mic_rotation_deg)),
                mic_centre[1] - (constants.inter_aural_mic_distance / 2) * np.sin(np.deg2rad(mic_rotation_deg)),
                constants.mic_height]
        right = [mic_centre[0] + (constants.inter_aural_mic_distance / 2) * np.cos(np.deg2rad(mic_rotation_deg)),
                 mic_centre[1] + (constants.inter_aural_mic_distance / 2) * np.sin(np.deg2rad(mic_rotation_deg)),
                 constants.mic_height]

        return np.round(np.c_[left, right], decimals=4)

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

    def source_location(self, distance, azimuth_deg):
        location = (self.room_dim / 2 + distance * np.r_[np.cos(np.deg2rad(azimuth_deg)),
                                                         np.sin(np.deg2rad(azimuth_deg)), 0.])
        return np.around(location, decimals=4)

    def plot_room(self, notebook=False):
        source_loc = self.source_location(distance=self.source_distance, azimuth_deg=self.source_azimuth)
        plt.grid()
        count = 0
        prediction_circle_list = []
        for i in self.rotation_list:
            print(count)
            mic_pos = self.mic_position(self.mic_centre, i)

            rectangle = plt.Rectangle((0, 0), self.room_dim[0], self.room_dim[1], fill=False, lw=3)
            plt.gca().add_patch(rectangle)

            dotted_line = plt.Line2D((mic_pos[0][0], mic_pos[0][1]), (mic_pos[1][0], mic_pos[1][1]),
                                     lw=5.,
                                     marker='.',
                                     markersize=20,
                                     markerfacecolor='r',
                                     markeredgecolor='r',
                                     alpha=1)
            plt.gca().add_line(dotted_line)

            polygon = plt.Polygon(self.get_triangle_points(self.mic_centre, mic_pos, scale_factor=4), color='g')
            plt.gca().add_patch(polygon)

            azimuth_line = plt.Line2D((self.mic_centre[0], source_loc[0]), (self.mic_centre[1], source_loc[1]), lw=1.,
                                      ls='-.', color='k')
            plt.gca().add_line(azimuth_line)

            circle = plt.Circle((source_loc[0], source_loc[1]), radius=0.1, fc='y')
            plt.gca().add_patch(circle)

            if count > 0:
                pred_loc_1 = self.source_location(self.source_distance, self.prediction_list[count - 1][0])
                pred_1_circle = plt.Circle((pred_loc_1[0], pred_loc_1[1]), radius=0.04, fc='k')
                prediction_circle_list.append(pred_1_circle)

                pred_loc_2 = self.source_location(self.source_distance, self.prediction_list[count - 1][1])
                pred_2_circle = plt.Circle((pred_loc_2[0], pred_loc_2[1]), radius=0.04, fc='k')

                prediction_circle_list.append(pred_2_circle)

            for j in prediction_circle_list:
                plt.gca().add_patch(j)

            plt.axis('scaled')
            plt.title('Room Configuration', fontsize=17)
            if count < len(self.prediction_list):
                count += 1
            else:
                pred_loc_ = self.source_location(self.source_distance, self.prediction)
                pred_circle = plt.Circle((pred_loc_[0], pred_loc_[1]), radius=0.08, fc='b')
                plt.gca().add_patch(pred_circle)

                at = AnchoredText("Pediction: {prediction} degrees".format(prediction=self.prediction),
                                  prop=dict(size=20), frameon=True,
                                  loc='upper left',
                                  )
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                self.fig.gca().add_artist(at)
            
#             plt.savefig("rotation_model_{count}".format(count=count), dpi=200)
            self.camera.snap()

        animation = self.camera.animate(interval=1200)

        if notebook:
            return HTML(animation.to_html5_video())
        else:
            animation.save('animation.mp4')
