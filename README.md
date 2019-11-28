# FER - Facial Expression Recognition

This work is to demonstrate the below problem: 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

A real time face detector and emotion classifier is built using Convolution Neural Network and OpenCV.
The CNN model is tuned for fine performance even on a low end device.

## Instructions
Follow the [guided tutorial](FER_CNN.ipynb) for neural network training.

Files Structure:
- FER_CNN.ipynb - Tutorial to train the CNN
- FER.py - Uses the pre-trained model to give inferences
- FER_osc.py - Uses the pre-trained model to give inferences / OSC
- model.json - Neural network architecture
- weights.h5 - Trained model weights
## Installation
Using Python virtual environment will be advisable.
* For model prediction

    `pip install -r requirements.txt`
    
    OR
    
    `pip install opencv-python`
    
    `pip install tensorflow` (Note here we are installing tensorflow-cpu)
    
    `pip install keras`
    
    `pip install python-osc` (for precessing example)
    
* For model training,
    `pandas` `numpy` `tensorflow` `keras` `matplotlib` `scikit-learn` `seaborn`
    
* Running the inference engine

Use the webcam

`python FER.py webcam <fps>`

Use a video file

`python FER.py <video_file_name> <fps>`

## Contributing
* Report issues on issue tracker
* Fork this repo
* Make awesome changes
* Raise a pull request

##
#### Copyright & License

Copyright (C) 2018  Mayur Madnani

Licensed under MIT License

See the [LICENSE](LICENSE).
