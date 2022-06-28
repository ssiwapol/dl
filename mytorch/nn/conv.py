# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        # Store input shape variables
        _, _, self.input_size = A.shape
        # Calculate output_size
        output_size = ((A.shape[-1] - self.kernel_size) // 1) + 1

        # Create Z with shape (batch_size, out_channels, output_size)
        Z = np.zeros((A.shape[0], self.out_channels, output_size))
        # Convolve W with A and store values in Z
        for i in range(output_size):
            Z[:,:, i] = np.tensordot(A[:, :, i:(i + self.kernel_size)], self.W, axes=((1,2),(1,2))) + self.b
            #Z[:,:, i] = np.einsum('ijk,ljk->il', A[:, :, i:i + self.kernel_size], self.W) + self.b
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # Sum dLdZ to dLdb with shape out_channels
        self.dLdb = np.sum(dLdZ, axis=(0,2))

        # Create dLdW with shape (out_channels, in_channels, kernel_size)
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))
        # Convole dLdZ with A and store values in dLdW
        for i in range(self.dLdW.shape[2]):
            self.dLdW[:, :, i] = np.einsum('ijk,ilk->lj', self.A[:,:, i:(i + dLdZ.shape[2])], dLdZ)
        
        # Flip weight
        W_f = np.flip(self.W, axis=2)
        # Pad dLdZ at axis=2
        pad_size = self.kernel_size - 1
        dLdZ_b = np.pad(dLdZ, ((0,0), (0,0), (pad_size,pad_size)), 'constant', constant_values=(0,0))
        # Create dLdA with shape (batch_size, in_channels, input_size)
        self.dLdA = np.zeros((dLdZ.shape[0], self.in_channels, self.input_size))
        # Convole flipped weight with padded dLdZ and store values in dLdA
        for i in range(self.dLdA.shape[2]):
            self.dLdA[:,:,i] = np.einsum('ijk,jlk->il', dLdZ_b[:,:,i:(i + W_f.shape[2])], W_f)

        return self.dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                            weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA1 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA1)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # Store variable A
        self.A = A
        # Store input shape variables
        self.batch_size, _, self.in_height, self.in_width = A.shape
        # Calculate output size
        self.out_height = ((self.in_height - self.kernel_size) // 1) + 1
        self.out_width = ((self.in_width - self.kernel_size) // 1) + 1
        
        # Create Z output with shape (n, out_channels, out_height, out_width)
        Z = np.zeros((self.batch_size, self.out_channels, self.out_height, self.out_width))
        # Convolve W with A and store in Z
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                Z[:,:,i,j] = np.einsum('ijkl,mjkl->im', self.A[:,:,i:(i + self.W.shape[2]),j:(j + self.W.shape[3])], self.W) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Sum dLdZ to dLdb with shape out_channels
        self.dLdb = np.sum(dLdZ, axis=(0,2,3))

        # Create dLdW with shape (out_channels, in_channels, kernel_size, kernel_size)
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        # Convole dLdZ with A and store in dLdW
        for i in range(self.dLdW.shape[2]):
            for j in range(self.dLdW.shape[3]):
                self.dLdW[:,:,i,j] = np.einsum('ijkl,imkl->mj', self.A[:,:,i:(i + dLdZ.shape[2]),j:(j + dLdZ.shape[3])], dLdZ)
        
        # Calculate padding size
        pad_size = self.kernel_size - 1
        # Pad dLdZ
        dLdZ_p = np.pad(dLdZ, ((0,0), (0,0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0,0))
        # Flip weight horizontally
        W_f = np.flip(self.W, axis=3)
        # Flip weight vertically
        W_f = np.flip(W_f, axis=2)
        # Create dLdA with shape (batch_size)
        dLdA = np.zeros((self.batch_size, self.in_channels, self.in_height, self.in_width))
         # Convole flipped weight with padded dLdZ and store values in dLdA
        for i in range(dLdA.shape[2]):
            for j in range(dLdA.shape[3]):
                dLdA[:,:,i,j] = np.einsum('ijkl,jmkl->im', dLdZ_p[:,:,i:(i + W_f.shape[2]),j:(j + W_f.shape[3])], W_f)

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, 
                                            weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdA1 = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA1)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                            weight_init_fn, bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)

        dLdA = self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, 
                                            weight_init_fn, bias_init_fn)
        self.upsample2d = Upsample2d(upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)

        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        self.batch_size, self.in_channel, self.out_channel = A.shape
        Z = A.reshape(self.batch_size, -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.batch_size, self.in_channel, self.out_channel)

        return dLdA

