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
def forwardpropagation(X):
    z1 = np.dot(X, W1) + b1
    a1 = ReLU(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2

def compute_loss(y_true, y_pred):
    epsilon = 1e-15  # small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip to prevent log(0) or log(1 - 0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

