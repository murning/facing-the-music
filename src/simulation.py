import numpy as np
import data_generator_lib
import pandas as pd
import librosa
import data_cnn_format as cnn
import gccphat
import constants
import rotate
import final_models
import utility_methods
from plotting import plotting


class Simulation:
    """
    Class for running a single simulation of doa estimation based on synthetic data
    """

    def __init__(self, directory, source_azimuth, source_distance, model):

        self.current_position = 0
        self.iteration = 0
        self.directory = directory
        self.predictions = []
        self.source_azimuth = source_azimuth
        self.source_distance = source_distance
        self.model = model
        self.rotation = 0
        self.mic_centre = np.array([1.5, 1.5])
        self.rotation_list = [0]
        self.prediction = 0
        self.initial_adjust = False
        self.current_model = self.get_model()  # init
        self.results = []
        self.audio_number = 0
        # TODO: move doa to here

    def store_prediction(self, doa_list):
        """
        convert relative prediction to home coordinates
        """

        true_doas = [utility_methods.cylindrical(self.current_position + doa_list[0]),
                     utility_methods.cylindrical(self.current_position + doa_list[1])]

        self.predictions.append(true_doas)

    def get_model(self):

        model = None
        if self.model == "gcc_cnn":
            model = final_models.gcc_cnn()
        elif self.model == "raw_cnn":

            model = final_models.raw_cnn()
        elif self.model == "raw_resnet":

            model = final_models.raw_resnet()
        elif self.model == "gcc_dsp":

            model = final_models.gcc_dsp()
        else:
            print("Error -> No file found")
        return model

    def simulated_record(self, mic_rotation):
        """
        Simulation of recording audio. Takes source position and mic rotation,
        simulates acoustics, and records to wav

        :param mic_rotation: angle of rotation of microphone to simulate the head movements


        """
        data_point = data_generator_lib.get_data(subset_size=1)[self.audio_number]
        wav_id, true_azimuth = data_generator_lib.generate_training_data(source_azimuth_degrees=self.source_azimuth,
                                                                         source_distance_from_room_centre=self.source_distance,
                                                                         SNR=-20,
                                                                         RT60=0.3,
                                                                         source=data_point,
                                                                         mic_centre=self.mic_centre,
                                                                         mic_rotation_degrees=mic_rotation,
                                                                         binaural_object=constants.room,
                                                                         dir=self.directory)

        output = [wav_id, true_azimuth, self.iteration]

        df = pd.DataFrame([output], columns=['stereo wav', 'True Azimuth', 'Rotation Iteration'])
        df.to_csv("{dir}/iteration_{iter}.csv".format(dir=self.directory, iter=self.iteration))

    def load_audio(self):
        """
        Reads wav file based on csv values, resamples audio to 8000hz, fixes length to 1 second
        :return: numpy array of stereo audio, DOA from file
        """
        df = pd.read_csv("{dir}/iteration_{iter}.csv".format(dir=self.directory, iter=self.iteration),
                         usecols=[1, 2, 3])

        doa_from_file = df.iloc[0][1]
        wav_name = df.iloc[0][0]
        filename = "{dir}/{wav_name}".format(dir=self.directory, wav_name=wav_name)

        y, sr = librosa.load(filename, mono=False)

        y_8k = librosa.resample(y, sr, 8000)
        result_x = librosa.util.fix_length(y_8k, 8000)

        return result_x, doa_from_file

    def format_raw_audio_cnn(self):
        """
            Format the stereo Audio file for input to the raw audio CNN
            :return: data formatted for raw audio CNN, DOA read from file
            """
        result_x, doa_from_file = self.load_audio()
        x = np.array([result_x])
        x_data = cnn.reshape_x_for_cnn(cnn.normalize_x_data(cnn.flatten_stereo(x)))

        return x_data, doa_from_file

    def format_gcc_cnn(self):
        """
            Format the stereo Audio file for input to the gcc_phat CNN
            :return: data formatted for gcc_phat CNN, DOA read from file
            """
        result_x, doa_from_file = self.load_audio()

        signal = result_x[0]
        reference_signal = result_x[1]
        _, raw_gcc_vector = gccphat.gcc_phat(signal=signal, reference_signal=reference_signal, fs=8000)

        cross_correlation_vector = cnn.reshape_x_for_cnn(cnn.normalize_x_data(np.array([raw_gcc_vector])))

        return cross_correlation_vector, doa_from_file

    def format_gcc_dsp(self):
        """
        Format stereo audio file for gcc_dsp model
        :return: signal, reference signal and doa_from_file
        """
        result_x, doa_from_file = self.load_audio()

        return result_x, doa_from_file

    def load_and_process_audio(self):
        """
        Wrapping loading and processing of models into a single function
        """
        output_vector = None
        doa = None
        if self.model == "gcc_cnn":
            output_vector, doa = self.format_gcc_cnn()
        elif self.model == "gcc_dsp":
            output_vector, doa = self.format_gcc_dsp()
        elif self.model == "raw_cnn":
            output_vector, doa = self.format_raw_audio_cnn()
        elif self.model == "raw_resnet":
            output_vector, doa = self.format_raw_audio_cnn()
        else:
            print("Error -> No file found")

        return output_vector, doa

    def compute_rotation(self):
        """
        compute rotation based on current and prior predictions
        :return:
        """
        if self.predictions[self.iteration][0] == 90.0 or self.predictions[self.iteration][0] == 270.0:
            self.rotation = 20
            self.initial_adjust = True
            return

        if self.iteration == 0 or (self.iteration == 1 and self.initial_adjust):
            self.rotation = rotate.get_90_deg_rotation(self.predictions[self.iteration])
        elif self.iteration == 1 or (self.iteration == 2 and self.initial_adjust):
            self.rotation = rotate.get_45_deg_rotation(self.predictions, self.current_position)
        elif self.iteration >= 2 or (self.iteration > 2 and self.initial_adjust):
            self.rotation = rotate.get_fine_rotation(self.iteration)

    def update_position(self):
        """
        update current position of microphone based on rotation
        :param rotation:
        :return:
        """
        self.current_position = utility_methods.cylindrical(self.current_position + self.rotation)

        self.rotation_list.append(self.current_position)

    def rotate_to_prediction(self):

        self.rotation = self.prediction - self.current_position - 90

    def simulate(self):

        while self.iteration < 6:

            self.simulated_record(mic_rotation=self.current_position)

            print("Recording Successful")

            vector, doa = self.load_and_process_audio()

            doa_list = self.current_model.predict(vector)

            print("Model Prediction: {list}".format(list=doa_list))

            self.store_prediction(doa_list)

            print("Prediction List: {list}".format(list=self.predictions))

            val = utility_methods.check_if_twice(self.predictions, self.iteration)

            if val is not None:
                self.prediction = val
                self.rotate_to_prediction()
                self.update_position()
                for i in range(4):
                    self.rotation_list.append(self.current_position)
                self.rotation_list.append(0)
                return self.prediction

            self.compute_rotation()

            print("Rotation: {rotation}".format(rotation=self.rotation))

            self.update_position()

            print("Current Position: {position}".format(position=self.current_position))

            self.iteration += 1

        self.prediction = utility_methods.get_mean_prediction(prediction_list=self.predictions)
        self.rotate_to_prediction()
        self.update_position()
        self.rotation_list.append(0)

        return self.prediction

    def evaluate(self, test_number):

        for i in np.arange(0, 355, 5):
            self.source_azimuth = i
            print("-------------------------------------------------------------------")
            print("Source Azimuth: {azimuth}".format(azimuth=self.source_azimuth))
            pred = self.simulate()
            print("final prediction: {pred}".format(pred=pred))
            print("-------------------------------------------------------------------")
            self.save_results()
            self.reset()

        df = pd.DataFrame(self.results, columns=['prediction', 'source_azimuth',
                                                 'Number of iterations',
                                                 'source distance'])

        df.to_csv("{dir}/base_test_{test_number}_{model}.csv".format(test_number=test_number,
                                                                     dir="evaluation_results",
                                                                     model=self.model))

    def evaluate_distance(self, test_number):

        azimuth = 0
        for i in np.arange(0.3, 1.5, 0.1):
            self.source_distance = i
            self.source_azimuth = azimuth
            print("Source Azimuth: {azimuth}".format(azimuth=self.source_azimuth))
            pred = self.simulate()
            print("final prediction: {pred}".format(pred=pred))
            print("-------------------------------------------------------------------")
            self.save_results()
            self.reset()
            azimuth += 35

        df = pd.DataFrame(self.results, columns=['prediction', 'source_azimuth',
                                                 'Number of iterations',
                                                 'source distance'])
        df.to_csv("{dir}/distance_test_{test_number}_{model}.csv".format(dir="evaluation_results", model=self.model,
                                                                         test_number=test_number))

    def save_results(self):

        results_row = [self.prediction, self.source_azimuth, self.iteration, self.source_distance]

        self.results.append(results_row)

    def reset(self):
        """
        This method resets the prediction, iteration, position and rotation values to initial state. Rotates The
        motor back to 0 degrees

        :return:
        """

        self.rotation = 0
        self.iteration = 0
        self.predictions = []
        self.prediction = 0
        self.current_position = 0
        self.rotation_list = [0]
        self.prediction = 0
        self.initial_adjust = False




if __name__ == '__main__':
    source_distance = 1  # source distance in meters
    source_azimuth = 10  # source azimuth in degrees

    # Instantiation of the simulation class
    sim = Simulation(directory="simulation_test",
                     source_azimuth=source_azimuth,
                     source_distance=source_distance,
                     model="raw_resnet")

    prediction = sim.simulate()  # Run simulation and get prediction

    plot = plotting(room_dim=constants.room_dim,
                    source_distance=source_distance,
                    source_azimuth=source_azimuth,
                    mic_centre=sim.mic_centre,
                    rotation_list=sim.rotation_list,
                    prediction_list=sim.predictions,
                    prediction=sim.prediction)

    plot.plot_room()

    print("Final Prediction: {prediction}".format(prediction=prediction))
