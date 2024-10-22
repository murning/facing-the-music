Facing The Music
==============================

## Binaural Sound Source Localisation Using Machine Learning

This repository serves to document my undergraduate thesis for a BSc in Electrical & Computer Engineering at the University of Cape Town. 
The goal of this project was to develop a binaural sound source localisation
system using machine learning. This was undertaken as a proof of concept for an auditory perception system in a search and rescue robot. 

The following research question was proposed: 

* In comparison to using classical signal processing techniques, can deep learning
  be used in the task of binaural sound source localisation in order to
  develop a system that is robust to noise and reverberation? 
  
* Furthermore, can the deployment of this system on a rotating robotic platform improve localisation accuracy?


The final deliverable of this project was a robotic system that was capable of locating a sound source and rotating to face it. All the models were run in real time on a Raspberry Pi 3 B+. 

Comprehensive documentation for this project can be found in my undergraduate ![thesis](./report/)




Video Demo
---------------

<p align="center">
  <img src=./images/demo.gif>
</p>

A full video demo of the project can be viewed [here](https://youtu.be/xl86-_YQZdM?t=185)

System Overview
---------------
<p align="center">
  <img src=./images/facing-the-overview.png>
</p>

 
Simulation
---------------

This gif shows a simulation of the final model in action. A data point with a
true direction of arrival of 70 degrees is synthesised and fed into the model.
Pairs of successive predictions are then plotted until the final prediction is
determined. In this instance the model correctly predicts a direction of
arrival of 70 degrees and turns to face the source.

<p align="center">
  <img src=./images/rotating.gif width="500" >
</p>

Overview 
---------------

* A data-synthesis method was designed and implemented to generate large datasets
  to be used for sound source localisation studies.

* Three deep learning models where trained and tested using these datasets.

* A signal processing method was implemented to act as a baseline with which the
  DL models could be compared.

* A rotation algorithm was concieved of to account for problems inherent in 360 degree
  binaural SSL.

* A simulation program was written to evaluate the aforementioned methods in
   software.
   
   
* A hardware system was designed and built with which the models were deployed.

* Comprehensive testing was performed in order to evaluate the system.




Findings
---------------

* Experimentation showed that even though a useful model of binaural sound
  source localisation was developed, its performance with respect to noise and reverberation
rejection did not show a significant improvement upon the existing signal processing
method. 

* This being said, the deep learning model was successfully deployed on a hardware
system and was able to perform direction of arrival estimation with promising accuracy
in a controlled environment. 

* One of the most promising findings was that the system, which was trained entirely on synthesized data, was capable of    performing robust localisation in real environments when deployed on the hardware. 

This work presents a starting point for further research in pursuit of a robust, end-to-end, data-driven solution to binaural sound source localisation.




Getting Started
---------------


If you wish to explore the work of this project, have a look through the jupyter
notebook [Facing the Music.ipynb](./src/notebooks/). This notebook
walks through the logic of the system through the following steps:

* Data Synthesis
* Data Preprocessing
* Simulation with deep learning models

If you would like to play around and explore the code and simulations, you can run the jupyter notebook on your own 
machine as follows. 

```
git clone https://github.com/murning/facing-the-music && cd facing-the-music
conda env create -n facing-the-music -f=facing-the-music.yml
conda activate facing-the-music
jupyter notebook ./src/notebooks/Facing\ The\ Music.ipynb
```




About
--------------

The following sections serve to provide a brief overview of the design process
as well as an introduction to the code base.




### Data

To train a network for binaural sound source localisation, a dataset of stereo
audio recordings with direction of arrival (DOA) labels was required. Unfortunately to
the best of my knowledge no such dataset exists. There are a number of datasets
that utilize a dummy head such as the
[KEMAR](https://www.gras.dk/products/head-torso-simulators-kemar/product/733-45bb),
but these are intentionally recorded to represent the acoustic properties of the
human torso and head. Furthermore, the existing datasets were reletively small
(< 500 data points), and thus in order to train deep NNs effectively, a larger
dataset was required.

To navigate this obstacle, a large dataset of labelled, stereo wav files was
synthesized using the [Image Source
Method](https://jontalle.web.engr.illinois.edu/uploads/537/Papers/Public/Allen/AllenBerkley79.pdf).
The implementation of this method was provided by the excellent
[pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/), a
python package for room acoustics simulation.

I wrote a class that creates an artificial room and binaural microphone
configuration. A mono audio recording from the Google Speech Commands dataset is
then "placed" in the room in a specified position. A stereo waveform is then generated with
the spatial information of the room and the direction of arrival of the source imparted on the original mono recording. Thus with the
stereo file, the direction of arrival of the sound can be easily perceived. An
audio demo of this is included in the [explanatory notebook](./src/notebooks/).

Example code to synthesize a single data point is shown below:

In this instance we have:
  - 4x4x4 meter room
  - reflection order of 17
  - microphone height of 2m
  - microphone centre of (x=2,y=2)
  - inter-microphone distance of 0.2m 
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
  <img src=./images/70_degree_room_config.png width="400">
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

Some example room configurations are shown below: 

Direction of Arrival: 350 degrees           |  Direction of Arrival: 120 degrees
:-------------------------:|:-------------------------:
![](./images/room_145.png)  |  ![](./images/room_120.png)

Synthesising such a large dataset is very computationally expensive.
Additionally, the dataset used in training these models is 50Gb+ in size, and
thus was not included in this repository. A parallel
implementation of the above code was created using python multiprocessing. This
was run on a GCP instance with 12 cores and 60GB of RAM. The script used to run
the data synthesis can be viewed [here](./src/data/generate_data.py). If you
would like to train a network using this data, you can 
synthesize a dataset of any size by configure the degrees of freedom in
`generate_data.py` and then running the script. For example:

``` python
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

```

and then run: 

``` shell
python generate_data.py
```


### Front-Back Labelling and the Cone of Confusion

#### Cone of Confusion

An interesting problem encountered in the case of a binaural microphone
configuration is the inherent ambiguity between the front and the back of the
listener. The time domain information received at each microphone does not allow
us to distinguish between a sound arriving from 70 degrees and -70 degrees as
illustrated below: 

<p align="center">
  <img src=./images/frontbackconfusion.png width="350">
</p>


To illustrate this, consider the following two room configurations: 

Direction of Arrival: 45 degrees           |  Direction of Arrival: 315 degrees
:-------------------------:|:-------------------------:
![](./images/room_45.png)  |  ![](./images/room_315.png)


The DOA can be computed by estimating
the time delay between each channel of the recorded signal. We can do this by
finding the peak of the cross correlation between the two signals. The cross
correlation technique used is the generalised cross-correlation with phase
transform (GCC-PHAT). Consider the following scenario:

GCC-PHAT: 45 degrees           |  GCC-PHAT: 315 degrees
:-------------------------:|:-------------------------:
![](./images/doa_45_gcc.png)  |  ![](./images/doa_315.png)

It is evident from the graphs that the peaks occur at the same sample delay for
the two different directions of arrival. Thus when estimating the direction of
arrival from the cross-correlation peaks, both result in 45 degrees. The
implementation of this computation can be viewed
[here](./src/models/gccphat.py). 


This is an effect that is experienced to some degree in the human auditory
system and is often referred to as the "cone of confusion". As humans we have developed a number
of novel ways of dealing with this issue, one of which being small head
movements that aid the process of localisation. Taking inspiration from this
biological phenomenon, this system utilizes a rotation algorithm in conjunction
with a series of predictions to mitigate front back ambiguity. 

#### Front-Back Labelling

For the purpose of training networks, during data synthesis each DOA was labelled
with the two possible directions that could arise as a result of the
front-back confusions. For example, a data point that has a DOA of 45 degrees would be labelled as 45 degrees and 315 degrees. This labelling
technique in conjunction with a rotation algorithm and successive
predictions allows for a probability for each possible DOA to be
determined. Thus the DOA with the highest probability is determined as the
true direction of arrival.

After data synthesis the file names and their associated directions of arrival
are stored in a file called `data_labels.csv`.

## Rotations

An activity diagram of the rotation algorithm is shown below. This illustrates
how successive predictions can be used to determine which of the two possible
directions the source is emanating from. The python implementation of this
algorithm can be seen in [simulation.py](./src/simulation/simulation.py)

<p align="center">
  <img src=./images/rotationmodel_diagram.png width="500">
</p>




## Models

Multiple models were designed. The architectures were based on the
ones proposed in this [paper](https://arxiv.org/pdf/1610.00087.pdf). The
following models were implemented and tested: 

| Model Name | Architecture | Input Vector    |
|------------|--------------|-----------------|
| m11_raw    | CNN          | Raw Audio       |
| m32        | ResNet       | Raw Audio       |
| m11_gcc    | CNN          | GCC-PHAT Vector |



The names of the models correspond to the different architectures outlined in
the aforementioned paper. The best performing model in this use case was the m11
CNN, but modified to use the GCC-PHAT vector as the input. The raw audio models
did not generalise well to real world conditions. Using the
GCC-PHAT of the two audio channels as input made the model far more robust to
changes in environment. The data preprocessing pipeline is shown below:

<p align="center">
  <img src=./images/pipeline_2.png> 
</p>

The following block diagram describes the direction of arrival estimation
pipeline in full:

<p align="center">
  <img src=./images/pipeline.png>
</p>

The m11_gcc model used the following architecture:

<p align="center">
  <img src=./images/gcc_architecture.png width="350" >
</p>


Tensorflow implementations of the models were based on code in this [repository](https://github.com/philipperemy/very-deep-convnets-raw-waveforms), and can be viewed in
[deep_learning_models.py](./src/models/deep_learning_models.py). To train any of
the models, run the following:

``` 
python deep_learning_models.py <model_choice> <model_name> <data_directory> <number_of_epochs>
```

For example: 

``` 
python deep_learning_models.py m11_gcc my_new_model wav_data 100 
```

## Hardware


The rotation algorithm and the various models were implemented on a Raspberry Pi
3 B+. Two microphones and a stepper motor were housed in a 3D printed case, the
design of which is shown in the image below: 
<p align="center">
    <img src=./images/hardwaredesign.png width="350" >
</p>


This housing was connected to the raspberry pi via a ribbon cable. An Apogee Duet
audio interface was used to record audio from the microphones. Detailed
instructions to replicate this hardware, including schematics, are included in
my thesis document.

Hardware Integration |  Microphones and stepper motor in housing | Full System |
:-------------------------:|:-------------------------:|:-------------------------:|
![](./images/hardware_connec.jpg)  |  ![](./images/hardware_integration.jpg)| ![](./images/hardwarefull.jpg) |



## Results

The results in presented in this section correspond with the tests defined in
section 7.2.2 of my thesis. This is a small subsection of the full set of results.
These results make use of the error circle for visualisation.
The error circle is a way of visualising the average error of a model in the azimuth plane.
The magnitude of the average error for each DOA is represented by it’s distance from
the origin. As discussed at length in my thesis, it is evident that deep learning can be used to
implement an effective binaural sound source localisation system. However, the results
also show that in comparison to a signal processing baseline, this model does not offer a
significant improvement in noise and reverberation robustness.

<p align="center">
    <img src=./images/results_1.png  >
</p>

<p align="center">
    <img src=./images/results_2.png  >
</p>

