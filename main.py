## Building Neural Network,

# Importing packages

import numpy as np

# Training settings
epochs = 100000
learning_rate = 0.01
loss_history = []

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(s):
    return s * (1 - s)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Using previous weights and bias (W1,b1, W2, b2)
def forward_pass(X,W1,B1,W2,B2):

    #Interaction between input and hidden layer 1-2
    Z1 = np.dot(X,W1) + B1
    # Activation function
    A1 = sigmoid(Z1)

    #Interaction between hidden and output layer
    Z2 = np.dot(A1,W2) + B2
    # Activation function
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2

# Backpropagation function
def backpropagation(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]  # number of samples

    # Step 1: Output layer error
    dZ2 = A2 - y             # shape: (4, 1)
    dW2 = np.dot(A1.T, dZ2) / m  # shape: (2, 1)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # shape: (1, 1)

    # Step 2: Hidden layer error
    dA1 = np.dot(dZ2, W2.T)          # shape: (4, 2)
    dZ1 = dA1 * sigmoid_derivative(A1)  # shape: (4, 2)
    dW1 = np.dot(X.T, dZ1) / m         # shape: (2, 2)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # shape: (1, 2)

    return dW1, db1, dW2, db2

# Random seed for consistent results
np.random.seed(42)

# Step 1 : setting up input and output data

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]) # 2 input feature
y = np.array([[0], [1], [1], [0]])

# Step 2 : initialize weight, from my understanding it's input layer and hidden layer
# W1 INPUT -> HIDDEN : Shape (2,2)
# 2 input features → 2 hidden neurons
W1 = np.random.randn(2,2)

# B1 (Bias for Hidden) : shape (1,2)
B1 = np.zeros((1,2))

# W2 (Hidden -> Output): shape (2, 1)
# 2 hidden neurons → 1 output
W2 = np.random.randn(2, 1)

# B2 (Bias for Output): shape (1, 1)
B2 = np.zeros((1, 1))
# I still have doubt how (4,2) + (1,2) matrix get added ?

# Step 3 : Forward Pass
Z1, A1, Z2, A2 = forward_pass(X, W1, B1, W2, B2)

loss = binary_cross_entropy(y, A2)
loss_history.append(loss)

print("Forward pass Done:",A1)
print(" \n Forward Pass Result:",A2)


for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_pass(X, W1, B1, W2, B2)
    loss = binary_cross_entropy(y, A2)
    loss_history.append(loss)

    dW1, db1, dW2, db2 = backpropagation(X, y, Z1, A1, Z2, A2, W2)

    # Update weights and biases
    W1 -= learning_rate * dW1
    B1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    B2 -= learning_rate * db2

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final output
_, _, _, y_hat = forward_pass(X, W1, B1, W2, B2)
print("\nFinal predictions:")
print(*y_hat)
print(*y)