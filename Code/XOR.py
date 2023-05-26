import numpy as np
from .module import *
from .Linear import *


class NN_XOR:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lin_layer = Linear(self.input_size, self.hidden_size)
        self.lin_layer2 = Linear(self.hidden_size, self.output_size)
        self.act_sig = Sigmoid()
        self.act_tan = TanH()
        self.loss_mse = MSELoss()

    def train(self, X_train, y_train, iteration, gradient_step):
        for _ in range(iteration):
            # Forward pass
            hidden_lin = self.lin_layer.forward(X_train)
            hidden_tan = self.act_tan.forward(hidden_lin)
            hidden_lin2 = self.lin_layer2.forward(hidden_tan)
            hidden_sig = self.act_sig.forward(hidden_lin2)
            loss = self.loss_mse.forward(y_train, hidden_sig)

            # Backward pass
            loss_back = self.loss_mse.backward(y_train, hidden_sig)
            delta_sig = self.act_sig.backward_delta(hidden_lin2, loss_back)
            delta_lin2 = self.lin_layer2.backward_delta(hidden_tan, delta_sig)
            delta_tan = self.act_tan.backward_delta(hidden_lin, delta_lin2)
            delta_lin = self.lin_layer.backward_delta(X_train, delta_tan)

            self.lin_layer2.backward_update_gradient(hidden_tan, delta_sig)
            self.lin_layer.backward_update_gradient(X_train, delta_tan)

            self.lin_layer2.update_parameters(gradient_step)
            self.lin_layer.update_parameters(gradient_step)

            self.lin_layer2.zero_grad()
            self.lin_layer.zero_grad()

    def predict(self, x):
        hidden_l = self.lin_layer.forward(x)
        hidden_l = self.act_tan.forward(hidden_l)
        hidden_l = self.lin_layer2.forward(hidden_l)
        hidden_l = self.act_sig.forward(hidden_l)
        return np.where(hidden_l >= 0.5, 1, 0)

