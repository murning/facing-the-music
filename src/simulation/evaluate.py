from simulation import Simulation

if __name__ == '__main__':
    model_name = ["gcc_dsp", "gcc_cnn", "raw_cnn", "raw_resnet"]

    for name in model_name:
        model = Simulation(directory="simulation_test", source_azimuth=0, source_distance=1, model=name)

        model.audio_number = 1
        for i in range(10):
            model.evaluate(i)
            model.audio_number += 1
