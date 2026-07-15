import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Trains logistic regression using Gradient Descent.
    X: Input features (samples, features)
    y: Target labels (samples,)
    """
    n_samples, n_features = X.shape
    
    # 1. Initialize weights (w) and bias (b)
    w = np.zeros(n_features)
    b = 0
    
    # 2. Training Loop
    for _ in range(steps):
        # Forward pass: Calculate predicted probability
        linear_model = np.dot(X, w) + b
        y_pred = _sigmoid(linear_model)
        
        # Calculate gradients (how much w and b contribute to the error)
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # Update weights and bias
        w -= lr * dw
        b -= lr * db
        
    return w, b
    # Write code here
    pass