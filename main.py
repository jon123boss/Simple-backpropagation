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

def backpropagation(X, y_true):
    global W1, b1, W2, b2, epsilon

    # Initialize gradients
    grad_W1 = np.zeros_like(W1)
    grad_W2 = np.zeros_like(W2)
    grad_b1 = np.zeros_like(b1)
    grad_b2 = np.zeros_like(b2)

    # Compute loss for current parameters
    y_pred = forwardpropagation(X)
    loss = compute_loss(y_true, y_pred)

    # Compute gradients using numerical differentiation
    # Gradient for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i, j] += epsilon
            y_pred = forwardpropagation(X)
            loss_new = compute_loss(y_true, y_pred)
            grad_W1[i, j] = (loss - loss_new) / epsilon
            W1[i, j] -= epsilon

    # Gradient for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i, j] += epsilon
            y_pred = forwardpropagation(X)
            loss_new = compute_loss(y_true, y_pred)
            grad_W2[i, j] = (loss - loss_new) / epsilon
            W2[i, j] -= epsilon

    # Gradient for b1
    for i in range(b1.shape[1]):
        b1[0, i] += epsilon
        y_pred = forwardpropagation(X)
        loss_new = compute_loss(y_true, y_pred)
        grad_b1[0, i] = (loss - loss_new) / epsilon
        b1[0, i] -= epsilon

    # Gradient for b2
    for i in range(b2.shape[1]):
        b2[0, i] += epsilon
        y_pred = forwardpropagation(X)
        loss_new = compute_loss(y_true, y_pred)
        grad_b2[0, i] = (loss - loss_new) / epsilon
        b2[0, i] -= epsilon

    return grad_W1, grad_W2, grad_b1, grad_b2

def train(X, y_true, epochs):
    global W1, b1, W2, b2

    for epoch in range(epochs):
        # Perform backpropagation and update weights and biases
        grad_W1, grad_W2, grad_b1, grad_b2 = backpropagation(X, y_true)

        W1 += learning_rate * grad_W1
        W2 += learning_rate * grad_W2
        b1 += learning_rate * grad_b1
        b2 += learning_rate * grad_b2

        # Compute and print loss every 100 epochs
        if (epoch+1) % 10000 == 0:
            y_pred = forwardpropagation(X)
            loss = compute_loss(y_true, y_pred)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

train(X, y_true, epochs=1000000)

while True:
    try:
        i_a = float(input("Input a: "))
        i_b = float(input("Input b: "))

        # Normalize input for prediction
        input_data = np.array([[i_a, i_b]])
        prediction = forwardpropagation(input_data)[0, 0]
        print(f"Prediction: {prediction:.4f}")

    except ValueError:
        print("Invalid input. Please enter a numeric value.")