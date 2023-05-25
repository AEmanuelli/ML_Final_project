from .module import Module

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class TanH(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)

    def update_parameters(self, learning_rate):
        pass  


class Sigmoide(Module):

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        return sigmoid(X)

    def backward_update_gradient(self, input, delta):
        pass  

    def backward_delta(self, input, delta):
        sig_X = sigmoid(input)
        return delta * sig_X * (1 - sig_X)

    def update_parameters(self, learning_rate):
        pass 
