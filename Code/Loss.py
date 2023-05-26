import numpy as np
class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass
    
class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2*(y-yhat)
    
    
    
    class CrossEntropyLoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return yhat - y

