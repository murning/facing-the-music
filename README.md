Facing The Music
==============================

## Binaural Sound Source Localisation Using Machine Learning

This repository serves to document my undergraduate thesis for a BscEng in Electrical & Computer Engineering at the University of Cape Town. 

The following research question was proposed: 

* In comparison to using classical signal processing techniques, can deep learning
  be used in the task of binaural sound source localisation in order to
  develop a system that is robust to noise and reverberation? 
  
* Furthermore, can the deployment of this system on a rotating robotic platform improve localisation accuracy?


The final deliverable of this project was a robotic system that was capable of locating a sound source and rotating to face it. All the models were run in real time on a Raspberry Pi 3 B+. 

Video Demo
---------------

<p align="center">
  <img src=./images/demo.gif>
</p>

A full video demo of the project can be viewed [here](https://youtu.be/xl86-_YQZdM?t=185)
 

About
--------------

The goal of this project was the implementation of a binaural sound source localisation
system using machine learning, with ultimate ends of deployment on a rotating robotic
platform. This was undertaken as a proof of concept for an auditory perception system in a search and rescue robot. 

### Data

To train a network for binaural sound source localisation, a dataset of stereo audio recordings with direction of arrival labels was required. Unfortunately to the best of my knowledge no such dataset exists. There are a number of datasets that utilize a dummy head such as the [KEMAR](https://www.gras.dk/products/head-torso-simulators-kemar/product/733-45bb), but these are intentionally recorded to represent the acoustic properties of the human torso and head. Furthermore, the existing datasets were reletively small (< 500 data points), and thus in order to train deep NNs effectively, a larger dataset was required. 

To navigate this obstacle, a large dataset of labelled, stereo wav files were synthesized using the [Image Source Method](https://jontalle.web.engr.illinois.edu/uploads/537/Papers/Public/Allen/AllenBerkley79.pdf). The implementation of this method was provided by the excellent [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/), a python package for room acoustics simulation.

I wrote a class that creates an artificial room and binaural microphone configuration. A mono audio recording from the Google Speech Commands dataset is then placed in the room in a specified position, arriving at the microphones with a specified direction of arrival. A stereo waveform is then generated with the spatial information imparted on the original mono recording. Thus with the setero file, the direction of arrival of the sound can be easily percieved. An audio demo of this is included in the [explanatory notebook](./src/notebooks/).

The code to synthesize a single data point is shown below: 

In this instance we have:
  - 4x4x4 meter room
  - reflection order of 17
  - microphone height of 2m
  - microphone centre of (x=2,y=2)
  - inter microphone distance of 0.2m 
  - a source azimuth of 70 degrees
  - an SNR of 0
  - an RT60 of 1 second (this corresponds to the reverb time of the room)

```
# Select an arbitrary data point from the Google Speech Commands dataset
source = data_generator_lib.get_data(1)[5] 

# Set up constant parameters for the room
room = Binaural(room_dim=np.r_[4., 4., 4.],
             max_order=17,
             speed_of_sound=343,
             inter_aural_distance=0.2,
             mic_height=2)

# Synthesize a stereo wav file with a direction of arrival of 70 degrees
room.generate_impulse_pair(source_azimuth_degrees=70,
                           source_distance_from_room_centre=1,
                           SNR=0,
                           RT60=1,
                           mic_centre=np.array([2, 2]),
                           mic_rotation_degrees=0,
                           fs=source.fs,
                           source_signal=source.data,
                           plot_room=True,
                           plot_impulse=False,
                           write_wav=True,
                           wav_name="demo_stereo_wav")
```

This code will generate a plot of the room configuration shown below. The yellow
dot is the mono audio source and the red dots are the microphones. The triangle
serves to represent the front of the microphone configuration (googly eyes).
<p align="center">
  <img src=./images/70_degree_room_config.png>
</p>


This data synthesis system allows for a number of degrees of freedom in
synthesizing a dataset. The final dataset used for training had over 300 000
data points varying in: 
- room size
- microphone position 
- source position
- mono audio source 
- signal to noise ratio
- reverb time

Synthesizing such a large dataset is very computationally expensive. A parallel
implementation of the above code was created using python multiprocessing. This
was run on a GCP instance with 12 cores and 60GB of RAM. 

### Design and implementation

* A data synthesis methodology was designed and implemented. This method was used to generate large synthetic datasets for use    in machine learning. 
* Classical signal processing techniques were used to act as a comparative
  baseline for the machine learning techniques utilized herein.
* Three deep learning models were designed, trained and evaluated. 
* These models were used in conjunction with a rotation
  algorithm implemented on the recording apparatus. The use of a rotation algorithm,
  in which predictions are updated with each movement, allows 360 ◦ localisation to take
  place. 

### Findings

* Experimentation showed that even though a useful model of binaural sound
  source localisation was developed, its performance with respect to noise and reverberation
rejection did not show a significant improvement upon the existing signal processing
method. 

* This being said, the deep learning model was successfully deployed on a hardware
system and was able to perform direction of arrival estimation with promising accuracy
in a controlled environment. 

* One of the most promising findings was that the system, which was trained entirely on synthesized data, was capable of    performing robust localisation in real environments when deployed on the hardware. 

This work presents a starting point for further research in pursuit of a robust, end-to-end, data-driven solution to binaural sound source localisation.

<p align="center">
  <img src=./images/rotating.gif width="512" height="512">
</p>

Getting Started
--------------------

If you wish to explore the work of this project, have a look throught the jupyter notebook Facing the Music. This notebook
walks through the logic of the system through the following steps:

* Data Synthesis
* Data Preprocessing
* Simulation

If you would like to play around and explore the code and simulations, you can run the jupyter notebook on your own 
machine as follows. 

```
git clone https://github.com/murning/facing-the-music && cd facing-the-music
conda env create -n facing-the-music -f=facing-the-music.yml
conda activate facing-the-music
jupyter notebook ./src/notebooks/Facing\ The\ Music.ipynb
```

Note that synthesizing the dataset is rather computationally expensive. If you plan to synthesize a dataset of 50000+ points you will need a lot of ram and as many cores as you can get your hands on. I used a GCP instance with 12 cores and 60gb of ram.

If you wish to explore the hardware design, please see the *Design* chapter in my ![thesis](./report/undergraduate_thesis_kevin_murning.pdf)


