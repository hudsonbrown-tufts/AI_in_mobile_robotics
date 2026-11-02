import numpy as np

# 1. Define 10 pairs of data points (X = features, y = binary targets)
X = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([2.0, 4.0, 6.0, 8.0])

# 2. Set hyperparameters
learning_rate = 0.1
epochs = 1000

# 3. Initialize parameters (weights)
m = 0.0  # Slope
b = 0.0  # Intercept

# 4. Gradient Descent Loop
for epoch in range(epochs):
    # Step A: Forward pass (Calculate predicted y)
    y = m * X + b
    
    # Step B: Calculate Gradients (The partial derivatives of loss)
    dm = np.mean((Y - y) * (-X))
    db = np.mean((Y - y)*(-1))
    
    # Step C: Update parameters by stepping down the gradient
    m -= learning_rate * dm
    b -= learning_rate * db

# 5. Output the finished equation parameters
print(f"Final Equation: y = {b:.4f} + {m:.4f}x")