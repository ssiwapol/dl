import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # Store input shape variables
        self.batch_size, self.in_channels, self.in_height, self.in_width = A.shape
        # Calculate output size
        out_height = ((self.in_height - self.kernel) // 1) + 1
        out_width = ((self.in_width - self.kernel) // 1) + 1

        # Create Z with shape (batch_size, in_channels, out_height, out_width)
        Z = np.zeros((self.batch_size, self.in_channels, out_height, out_width))
        # Create max_index to store data
        self.max_i = np.zeros_like(Z)
        self.max_j = np.zeros_like(Z)
        # Loop to find the maximum of each convolution
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                a = A[:, :, i:(i + self.kernel), j:(j + self.kernel)]
                Z[:, :, i, j] = np.max(a, axis=(2,3))
                ind = np.argmax(a.reshape(a.shape[0], a.shape[1], -1), axis=2)
                ind = np.unravel_index(ind, (a.shape[2], a.shape[3]))
                self.max_i[:, :, i, j] = ind[0] + i
                self.max_j[:, :, i, j] = ind[1] + j
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Create dLdA with shape (batch_size, in_channels, in_height, in_width)
        dLdA = np.zeros((self.batch_size, self.in_channels, self.in_height, self.in_width))
        # Loop update dLdA with value from dLdZ at max index
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        dLdA[i, j, int(self.max_i[i,j,k,l]), int(self.max_j[i,j,k,l])] += dLdZ[i,j,k,l]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # Store input shape variables
        self.batch_size, self.in_channels, self.in_height, self.in_width = A.shape
        # Calculate output size
        self.out_height = ((self.in_height - self.kernel) // 1) + 1
        self.out_width = ((self.in_width - self.kernel) // 1) + 1

        # Create Z with shape (batch_size, in_channels, out_height, out_width)
        Z = np.zeros((self.batch_size, self.in_channels, self.out_height, self.out_width))
        # Loop to calculate mean of each convolution
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                a = A[:, :, i:(i + self.kernel), j:(j + self.kernel)]
                Z[:, :, i, j] = np.average(a, axis=(2,3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Create dLdA with shape (batch_size, in_channels, in_height, in_width)
        dLdA = np.zeros((self.batch_size, self.in_channels, self.in_height, self.in_width))
        # Loop update dLdA with dLdZ
        for i in range(dLdA.shape[2] - self.out_height + 1):
            for j in range(dLdA.shape[3] - self.out_width + 1):
                dLdA[:,:,i:(i+self.out_height),j:(j+self.out_width)] += dLdZ / (self.kernel*self.kernel)
        
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z1)
        
        return Z 
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA1 = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA1)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA1 = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA1)

        return dLdA
