import numpy as np

# Hyperparameters
input_size = 2    # Number of input neurons
hidden_size = 2   # Number of hidden neurons
output_size = 1   # Number of output neurons
learning_rate = 0.03
epsilon = 0.0001    # Small value for numerical gradient estimation

# Example usage with dummy data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])

# Weights and biases initialization
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(1, output_size)

def ReLU(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Setting up starting functions