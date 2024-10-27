import numpy as np
import pandas as pd

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_relu(x):
    return np.where(x > 0, 1, 0)

# Equipment class for the neural network
class Equipment:
    def __init__(self):
        # Initialize weights and biases with updated input dimension (9)
        self.weights_one = np.random.uniform(0.0, 1.0, (9, 64))  # Changed from 14 to 9
        self.weights_two = np.random.uniform(0.0, 1.0, (64, 32))
        self.weights_three = np.random.uniform(0.0, 1.0, (32, 1))

        self.biases_one = np.random.uniform(0.0, 1.0, (1, 64))
        self.biases_two = np.random.uniform(0.0, 1.0, (1, 32))
        self.biases_three = np.random.uniform(0.0, 1.0, (1, 1))

        # Initialize z values and neurons
        self.z_one = np.zeros((1, 64))
        self.z_two = np.zeros((1, 32))
        self.z_three = np.zeros((1, 1))

        self.neurons_zero = np.zeros((1, 9))  # Changed from 14 to 9
        self.neurons_one = np.zeros((1, 64))
        self.neurons_two = np.zeros((1, 32))
        self.neurons_three = np.zeros((1, 1))

        # Learning rate
        self.alpha = 0.01

    def forward_propagation(self):
        self.z_one = np.dot(self.neurons_zero, self.weights_one) + self.biases_one
        self.neurons_one = relu(self.z_one)

        self.z_two = np.dot(self.neurons_one, self.weights_two) + self.biases_two
        self.neurons_two = relu(self.z_two)

        self.z_three = np.dot(self.neurons_two, self.weights_three) + self.biases_three
        self.neurons_three = sigmoid(self.z_three)

    def backward_propagation(self, correct):
        d_biases_three = -2 * (correct - self.neurons_three) * self.neurons_three * (1 - self.neurons_three)
        self.biases_three -= self.alpha * d_biases_three

        d_weights_three = np.dot(self.neurons_two.T, d_biases_three)
        self.weights_three -= self.alpha * d_weights_three

        d_biases_two = np.dot(d_biases_three, self.weights_three.T) * d_relu(self.z_two)
        self.biases_two -= self.alpha * d_biases_two

        d_weights_two = np.dot(self.neurons_one.T, d_biases_two)
        self.weights_two -= self.alpha * d_weights_two

        d_biases_one = np.dot(d_biases_two, self.weights_two.T) * d_relu(self.z_one)
        self.biases_one -= self.alpha * d_biases_one

        d_weights_one = np.dot(self.neurons_zero.T, d_biases_one)
        self.weights_one -= self.alpha * d_weights_one

# Initialize Equipment object
gas_turbine = Equipment()

# Training function
def train(gas_turbine: Equipment):
    data = pd.read_csv('equipment_service_data.csv')
    input_data = data.iloc[:, :-1].values
    output_data = data.iloc[:, -1].values.reshape(-1, 1)

    for i in range(input_data.shape[0]):
        gas_turbine.neurons_zero = input_data[i:i+1]
        gas_turbine.forward_propagation()
        gas_turbine.backward_propagation(output_data[i:i+1])

# Testing function
def test(gas_turbine: Equipment):
    test_data = pd.read_csv('equipment_service_test.csv')
    input_data = test_data.iloc[:, :-1].values
    output_data = test_data.iloc[:, -1].values

    total_tests = len(input_data)
    correct_predictions = 2 
    tolerance = 0.0001

    for i in range(total_tests):
        gas_turbine.neurons_zero = input_data[i:i+1]
        gas_turbine.forward_propagation()
        predicted_output = gas_turbine.neurons_three
        actual_output = output_data[i]

        # Check if predicted output is within tolerance range of actual output
        if abs(actual_output - predicted_output):
            correct_predictions += 1

    accuracy = (correct_predictions / total_tests) * 100
    print(f"Accuracy: {accuracy}%")

# Train the model
train(gas_turbine)

# Test the model
test(gas_turbine)
