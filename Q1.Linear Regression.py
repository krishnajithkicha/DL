import numpy as np
import time
from sklearn.datasets import load_diabetes

# ----------------------------
# Load Diabetes Dataset
# ----------------------------
diabetes = load_diabetes()

X = diabetes.data      # Input features
y = diabetes.target    # Output

# Add bias column
X = np.c_[np.ones(X.shape[0]), X]

print("Dataset : Diabetes")
print("Samples :", X.shape[0])
print("Features:", X.shape[1] - 1)
print()

# ----------------------------
# Loss Function
# ----------------------------
def loss(X, y, theta):
    prediction = X @ theta
    return np.mean((prediction - y) ** 2)

# ----------------------------
# Batch Gradient Descent
# ----------------------------
def batch_gd(X, y):
    rows, cols = X.shape

    theta = np.zeros(cols)

    learning_rate = 0.1
    epochs = 1000
    updates = 0

    start = time.time()

    for i in range(epochs):

        prediction = X @ theta

        gradient = (X.T @ (prediction - y)) / rows

        theta = theta - learning_rate * gradient

        updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)

# ----------------------------
# Stochastic Gradient Descent
# ----------------------------
def sgd(X, y):
    rows, cols = X.shape

    theta = np.zeros(cols)

    learning_rate = 0.01
    epochs = 1000
    updates = 0

    start = time.time()

    for e in range(epochs):

        for i in range(rows):

            prediction = X[i] @ theta

            gradient = (prediction - y[i]) * X[i]

            theta = theta - learning_rate * gradient

            updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)

# ----------------------------
# Mini Batch Gradient Descent
# ----------------------------
def mini_batch_gd(X, y):
    rows, cols = X.shape

    theta = np.zeros(cols)

    learning_rate = 0.05
    epochs = 1000
    batch_size = 64
    updates = 0

    start = time.time()

    for e in range(epochs):

        for i in range(0, rows, batch_size):

            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            prediction = X_batch @ theta

            gradient = (X_batch.T @ (prediction - y_batch)) / len(X_batch)

            theta = theta - learning_rate * gradient

            updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)

# ----------------------------
# Train the Models
# ----------------------------
batch = batch_gd(X, y)
stochastic = sgd(X, y)
mini = mini_batch_gd(X, y)

# ----------------------------
# Display Results
# ----------------------------
print("-----------------------------------------------------------")
print("Optimizer\tTime(s)\tUpdates\t\tFinal Loss")
print("-----------------------------------------------------------")

print(f"Batch GD\t{batch[0]:.4f}\t{batch[1]}\t\t{batch[2]:.2f}")

print(f"SGD\t\t{stochastic[0]:.4f}\t{stochastic[1]}\t\t{stochastic[2]:.2f}")

print(f"Mini Batch\t{mini[0]:.4f}\t{mini[1]}\t\t{mini[2]:.2f}")
