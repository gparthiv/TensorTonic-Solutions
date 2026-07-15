**LOGISTIC REGRESSION**

- **Prediction:** The model uses the **Sigmoid function** to turn raw input scores into a probability between 0 and 1 in def _sigmoid(z)
- **Evaluation:** We compare this probability to the actual target (0 or 1) using a **Loss Function** (Binary Cross-Entropy). We aim to keep this value as low as possible.
- **Optimization:** We use **Gradient Descent** to iteratively update the model's weights (w) and bias (b). By calculating the gradient (the direction of steepest increase in error), we move in the opposite direction to shrink the loss