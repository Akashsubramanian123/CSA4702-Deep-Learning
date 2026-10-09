import numpy as np
 
class LinearNeuron:
    """Single-layer neural network: Y_hat = X . W + b"""
    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (n_features, 1))   # weight matrix (3 x 1)
        self.b = np.zeros((1,))                        # bias vector
 
    def forward(self, X):
        # matrix multiplication (m x 3)(3 x 1) -> (m x 1), then bias added to every row
        return X @ self.W + self.b
 
    def train(self, X, y, lr=0.05, epochs=300):
        m = X.shape[0]
        history = []
        for _ in range(epochs):
            y_hat = self.forward(X)
            error = y_hat - y
            history.append(float(np.mean(error ** 2)))
            dW = (2 / m) * X.T @ error
            db = (2 / m) * error.sum(axis=0)
            self.W -= lr * dW
            self.b -= lr * db
        return history
