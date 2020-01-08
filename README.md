Facing The Music
==============================

## Binaural Sound Source Localisation Using Machine Learning

This repository serves to document my undergraduate thesis for a BscEng in Electrical & Computer Engineering at the University of Cape Town. 

The following research question was proposed: 

* In comparison to using classical signal processing techniques, can deep learning
  be used in the task of binaural sound source localisation in order to
  develop a system that is robust to noise and reverberation? 
  
* Furthermore, can the deployment of this system on a rotating robotic platform improve localisation accuracy?


The final deliverable of this project was a robotic system that was capable of locating a sound source and rotating to face it. All research performed in this project is reproducible through the [explanatory notebooks](./notebooks/).

Video Demo
---------------

<p align="center">
  <img src=./images/demo.gif>
</p>

A full video demo of the project can be viewed here https://youtu.be/xl86-_YQZdM?t=185
 

About
--------------

The goal of this project was the implementation of a binaural sound source localisation
system using machine learning, with ultimate ends of deployment on a rotating robotic
platform. 

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



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── report             <- Undergraduate Thesis pdf
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download and generate data
    │   │   
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │   
    │   │   
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

