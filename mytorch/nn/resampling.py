import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # Expand array by create repeated array
        Z = np.repeat(A, self.upsampling_factor, axis=-1)
        # Restrict array to last index
        Z = Z[:,:,:-(self.upsampling_factor - 1)] if self.upsampling_factor > 1 else Z
        # Create temporary zero array with the same size
        a = np.zeros_like(Z)
        # Set the value to 1 at step = upsampling_factor
        a[:,:,::self.upsampling_factor] = 1
        # Multiply two array by element-wise to get the result
        Z = Z * a

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        # Select array at upsampling_factor step
        dLdA = dLdZ[:,:,::self.upsampling_factor]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # Store input shape variables
        _, _, self.input_width = A.shape
        # Select array at downsampling step
        Z = A[:,:,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        # Expand array by create repeated array
        dLdA = np.repeat(dLdZ, self.downsampling_factor, axis=-1)
        # Decrease the size of array equals to input array
        dLdA = dLdA[:,:,:self.input_width]
        # Create temporary zero array with the same size
        a = np.zeros_like(dLdA)
        # Set the value to 1 at step = downsampling_factor
        a[:,:,::self.downsampling_factor] = 1
        # Multiply two array by element-wise to get the result
        dLdA = dLdA * a

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # Expand array by create repeated array in 2 diminsions
        Z = np.repeat(A, self.upsampling_factor, axis=-1)
        Z = np.repeat(Z, self.upsampling_factor, axis=-2)
        # Restrict array to last index of each dimension
        Z = Z[:,:,:-(self.upsampling_factor - 1),:-(self.upsampling_factor - 1)] if self.upsampling_factor > 1 else Z
        # Create temporary zero array with the same size
        a = np.zeros_like(Z)
        # Set the value to 1 at step = upsampling_factor
        a[:,:,::self.upsampling_factor,::self.upsampling_factor] = 1
        # Multiply two array by element-wise to get the result
        Z = Z * a

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

         # Select array at upsampling_factor step for both dimensions
        dLdA = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # Store input shape variables
        _, _, self.input_height, self.input_width = A.shape
        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Expand array by create repeated array in 2 diminsions
        dLdA = np.repeat(dLdZ, self.downsampling_factor, axis=-1)
        dLdA = np.repeat(dLdA, self.downsampling_factor, axis=-2)
        # Decrease array size equals to input array
        dLdA = dLdA[:,:,:self.input_height,:self.input_width]
        # Create temporary zero array with the same size
        a = np.zeros_like(dLdA)
        # Set the value to 1 at step = downsampling_factor
        a[:,:,::self.downsampling_factor,::self.downsampling_factor] = 1
        # Multiply two array by element-wise to get the result
        dLdA = dLdA * a

        return dLdA