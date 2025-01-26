import numpy as np
from typing import List, Tuple

class SingleNeuron:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, tolerance: float = 0.0001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.error_history = []
        self.weight_history = []
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        sx = self.sigmoid(x)
        return sx * (1 - sx)
    
    def initialize_weights(self, n_features: int) -> None:
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> Tuple[List[float], List[np.ndarray]]:
        self.initialize_weights(X.shape[1])
        
        for epoch in range(self.epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Calculate error
            error = np.mean((y_pred - Y) ** 2)
            self.error_history.append(error)
            
            # Save weights history
            self.weight_history.append(np.copy(self.weights))
            
            # Check for convergence
            if error < self.tolerance:
                break
            
            # Backward pass
            delta = (y_pred - Y) * self.sigmoid_derivative(z)
            
            # Update weights and bias
            self.weights -= self.learning_rate * np.dot(X.T, delta)
            self.bias -= self.learning_rate * np.mean(delta)
        
        return self.error_history, self.weight_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
