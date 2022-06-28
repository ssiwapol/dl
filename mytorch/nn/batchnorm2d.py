import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = (self.BW * NZ) + self.Bb
            return BZ

        self.Z = Z
        self.N = Z.shape[0] * Z.shape[2] * Z.shape[3]  # TODO

        self.M = (1/self.N) * np.einsum('ijkl->j', Z)  # TODO
        self.V = (1/self.N) * np.einsum('ijkl->l', np.square(np.transpose(self.Z, (0, 3, 2, 1)) - self.M))  # TODO
        self.NZ = np.transpose((np.transpose(self.Z, (0, 3, 2, 1)) - self.M) / np.sqrt(self.V + self.eps), (0, 3, 2, 1))  # TODO
        self.BZ = (self.BW * self.NZ) + self.Bb  # TODO

        self.running_M = (self.alpha * self.running_M) + ((1 - self.alpha) * self.M).reshape(1, -1, 1, 1)  # TODO
        self.running_V = (self.alpha * self.running_V) + ((1 - self.alpha) * self.V).reshape(1, -1, 1, 1) # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.einsum('ijkl->j', dLdBZ * self.NZ).reshape(1, -1, 1, 1)  # TODO
        self.dLdBb = np.einsum('ijkl->j', dLdBZ).reshape(1, -1, 1, 1)  # TODO

        dLdNZ = np.einsum('ijkl, mjmm->ijkl', dLdBZ, self.BW)  # TODO

        dLdV = dLdNZ
        dLdV = dLdV * np.transpose((np.transpose(self.Z, (0, 3, 2, 1)) - self.M), (0, 3, 2, 1))
        dLdV = np.einsum('ijkl, j->ijkl', dLdV, np.power(self.V + self.eps, -(3/2)))
        dLdV = (-1/2) * np.einsum('ijkl->j', dLdV)


        dLdM1 = np.einsum('ijkl, j->ijkl', dLdNZ, np.power(self.V + self.eps, (-1/2)))
        dLdM1 = -np.einsum('ijkl->j', dLdM1)

        dLdM2 = np.transpose(self.Z, (0, 3, 2, 1)) - self.M
        dLdM2 = np.einsum('ijkl->l', dLdM2)
        dLdM2 = - (2/self.N) * dLdV * dLdM2
        
        dLdM = dLdM1 + dLdM2


        dLdZ1 = np.einsum('ijkl, j->ijkl', dLdNZ, np.power(self.V + self.eps, (-1/2)))
        dLdZ2 = (2/self.N) * np.einsum('ijkl, l->ilkj', (np.transpose(self.Z, (0, 3, 2, 1)) - self.M), dLdV)
        dLdZ3 = (dLdM * (1/self.N)).reshape(1, -1, 1, 1)
        dLdZ = dLdZ1 + dLdZ2 + dLdZ3

        return dLdZ
