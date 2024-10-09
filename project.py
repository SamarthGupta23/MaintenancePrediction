import numpy as np
import random
import math

def relu(x):
    return np.maximum(0, x)  # Use numpy's max for element-wise operation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Use numpy for element-wise exponential

def d_relu(x):
    return np.where(x > 0, 1, 0)  # Use numpy to handle arrays

class Equipment:
    def __init__(self):
        # Initialize weights with random values
        self.weights_one = np.random.uniform(0.0, 1.0, (14, 64))  # Weights from input to first hidden layer
        self.weights_two = np.random.uniform(0.0, 1.0, (64, 32))   # Weights from first to second hidden layer
        self.weights_three = np.random.uniform(0.0, 1.0, (32, 1))  # Weights from second hidden layer to output

        # Initialize biases
        self.biases_one = np.random.uniform(0.0, 1.0, (1, 64))  # Correct shape for first layer
        self.biases_two = np.random.uniform(0.0, 1.0, (1, 32))   # Correct shape for second layer
        self.biases_three = np.random.uniform(0.0, 1.0, (1, 1))  # Correct shape for output layer

        # Initialize z values
        self.z_one = np.zeros((1, 64))  # Shape matching first hidden layer
        self.z_two = np.zeros((1, 32))   # Shape matching second hidden layer
        self.z_three = np.zeros((1, 1))  # Shape matching output layer

        # Initialize neurons
        self.neurons_zero = np.zeros((1, 14))
        self.neurons_one = np.zeros((1, 64))
        self.neurons_two = np.zeros((1, 32))
        self.neurons_three = np.zeros((1, 1))

        # Initialize learning rate
        self.alpha = 0.01

    def forward_propagation(self):
        # Forward propagation using class attributes
        self.z_one = np.dot(self.neurons_zero, self.weights_one) + self.biases_one
        self.neurons_one = relu(self.z_one)

        self.z_two = np.dot(self.neurons_one, self.weights_two) + self.biases_two
        self.neurons_two = relu(self.z_two)

        self.z_three = np.dot(self.neurons_two, self.weights_three) + self.biases_three
        self.neurons_three = sigmoid(self.z_three)

    def backward_propagation(self, correct):
        # Updating biases and weights for the output layer
        d_biases_three = -2 * (correct - self.neurons_three) * self.neurons_three * (1 - self.neurons_three)
        self.biases_three -= self.alpha * d_biases_three

        d_weights_three = np.dot(self.neurons_two.T, d_biases_three)  # Correct shape
        self.weights_three -= self.alpha * d_weights_three

        # Updating biases and weights for the second hidden layer
        d_biases_two = np.dot(d_biases_three, self.weights_three.T) * d_relu(self.z_two)
        self.biases_two -= self.alpha * d_biases_two

        d_weights_two = np.dot(self.neurons_one.T, d_biases_two)  # Correct shape
        self.weights_two -= self.alpha * d_weights_two

        # Updating biases and weights for the first hidden layer
        d_biases_one = np.dot(d_biases_two, self.weights_two.T) * d_relu(self.z_one)
        self.biases_one -= self.alpha * d_biases_one

        d_weights_one = np.dot(self.neurons_zero.T, d_biases_one)  # Correct shape
        self.weights_one -= self.alpha * d_weights_one


def train():
    gas_turbine = Equipment()
    gas_turbine.__innit__() # initialising variables
    # Define actual input and output arrays (replace these with your data)
    input_data = np.random.rand(100, 14)  # Example: 100 samples of 14 features
    output_data = np.random.rand(100, 1)  # Example: 100 samples of output

    for i in range(input_data.shape[0]):  # Iterate over each input sample
        gas_turbine.neurons_zero = input_data[i:i+1]  # Set the input
        gas_turbine.forward_propagation()  # Forward pass
        gas_turbine.backward_propagation(output_data[i:i+1])  # Backward pass
        
        accuracy = abs(gas_turbine.neurons_three - output_data[i:i+1]) / output_data[i:i+1]
        print(f"Accuracy: {accuracy}")

# Call the train function to execute
train()
