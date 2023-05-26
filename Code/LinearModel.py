import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from .Loss import *
from .Linear import *

class LinearModel:
    def __init__(self):
        self.linear_model = None
        self.mse_loss = None
        self.loss_history = []
        self.accuracy_history = []

    def train(self, X, y, epochs=100, learning_rate=1e-4):
        n_features = X.shape[1]
        self.linear_model = Linear(n_features, 1)
        self.mse_loss = MSELoss()

        for epoch in range(epochs):
            y_pred = self.linear_model.forward(X)
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            loss = self.mse_loss.forward(y.reshape(-1, 1), y_pred)
            accuracy = accuracy_score(y, np.round(y_pred.flatten()))
            self.loss_history.append(loss.mean())
            self.accuracy_history.append(accuracy)

            delta = self.mse_loss.backward(y.reshape(-1, 1), y_pred)
            self.linear_model.backward_update_gradient(X, delta)
            self.linear_model.update_parameters(learning_rate=learning_rate)
            self.linear_model.zero_grad()

    def predict(self, X):
        if self.linear_model is None:
            raise Exception("Model not trained yet. Please call 'train' method first.")
        return np.round(self.linear_model.forward(X))

    def plot_loss_history(self):
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()

    def plot_accuracy_history(self):
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.show()

