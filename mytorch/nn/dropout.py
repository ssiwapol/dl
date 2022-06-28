import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # Generate mask and apply to x
            self.mask = 1 - (np.random.binomial(n=1, p=self.p, size=x.shape))
            #self.mask = np.random.rand(*x.shape) < self.p
            z = (self.mask * x) / (1 - self.p)
            return z
            
        else:
            return x
		
    def backward(self, delta):
        # Multiply mask with delta and return
        delta = self.mask * delta
        return delta
