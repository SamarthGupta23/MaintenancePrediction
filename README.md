# Equipment Maintenance Prediction Model

## Overview

This project implements a neural network for predicting equipment maintenance needs based on various input features. The model is designed to assist in maintenance prediction for gas turbines, focusing on factors that affect operational efficiency and reliability.

## Features

- **Input Layer**: The model uses 9 input neurons, which capture critical features affecting equipment performance. These features include but are not limited to:
  - Work Hours After Last Service
  - Equipment Condition
  - Operating Environment (e.g., Dust, Temperature, Humidity)
  - Maintenance Actions (e.g., Lubrication, Component Replacement)

- **Neural Network Architecture**: The model consists of:
  - An input layer with 9 neurons.
  - One hidden layer with 64 neurons using the ReLU activation function.
  - A second hidden layer with 32 neurons, also using ReLU activation.
  - An output layer with 1 neuron that uses the sigmoid activation function to predict maintenance needs.

- **Training Process**: The model is trained using data from equipment maintenance records. Notably, this project was implemented **from scratch** without using any external libraries. The backpropagation algorithm was manually coded to update the weights and biases of the network, utilizing a learning rate of 0.01.

- **Testing and Evaluation**: The model's performance is evaluated on a separate test dataset. The accuracy of the model is a key metric that was calculated to be **88%**.

## Implementation

### Dependencies
This project is built entirely from scratch, and therefore, **no external libraries** are used. The implementation relies solely on **NumPy** for basic array manipulations, which can be easily replaced by manual operations if needed.
