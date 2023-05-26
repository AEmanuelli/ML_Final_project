import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.include_bias = bias
        self.init_params()

    def init_params(self):
        self._parameters = np.random.randn(self.input_size + int(self.include_bias), self.output_size)
        self._gradient = np.zeros_like(self._parameters)

    def forward(self, X):
        assert X.shape[1] == self.input_size, ValueError("X must be of shape (batch_size, input_size)")

        weights = self._parameters
        if self.include_bias:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((X, ones))
        self.output = np.matmul(X, weights)

        return self.output

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        if self.include_bias:
            ones = np.ones((input.shape[0], 1))
            input = np.hstack((input, ones))
        
        self._gradient += np.matmul(input.T, delta)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        weights = self._parameters
        d_out = np.matmul(delta, weights[:-1].T) if self.include_bias else np.matmul(delta, weights.T)
        return d_out

    def zero_grad(self):
        self._gradient = np.zeros_like(self._parameters)
        
        
   def update_parameters(self, learning_rate=0.001):
         self._parameters -= learning_rate * self._gradient
