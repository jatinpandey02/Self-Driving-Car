# Self-Driving-Car
Steering wheel prediction model 

# Objective
Our aim is to predict the steering angle for a car. The model predicts the angle frame by frame.

# Tools Used
The task was accomplished using tensorflow-keras on Google Colab. Later the weights were loaded on the local machine to make frame by frame video of the dashboard cam and predicted steering angle using openCV.

# Dataset used
https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view 

# Model Architecture
The model was a convolutional neural network with a few Dense layer at the end. 'Relu" activation layer was used at all layers except the last one where 'tanh' was used as the model would have to predict both negative and positive values of steering angle. 

# Final results
![](selfdrivingcar.gif)
