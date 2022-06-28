import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        if eval:
          return x

        bi_ch = 1 - (np.random.binomial(n=1, p=self.p, size=(x.shape[0], x.shape[1])))
        self.mask = np.einsum('ij,ijkl->ijkl', bi_ch, np.ones_like(x))
        z = (self.mask * x) / (1 - self.p)
        return z

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """

        delta = (self.mask * delta) / (1-self.p)
        return delta
