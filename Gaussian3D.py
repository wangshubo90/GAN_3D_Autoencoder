import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.stats as st


def _gaussianMatrix(sigma, kernel_size):
    
    """
    creates gaussian kernel with side length kernel_size and a sigma 
    """

    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy, zz= np.meshgrid(ax, ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.power(sigma,3))

    return kernel / np.sum(kernel)

@tf.function    
def Gaussian3DFilter(input, sigma, kernel_size=None):
    """
    apply 3d guassian filter on a 3d image (a 5D tf.tensor)
    """
    if kernel_size == None:
        kernel_size = int(sigma * 6)+1

    kernel = _gaussianMatrix(sigma, kernel_size)

    kernel = tf.constant(kernel, dtype=tf.float32)
    kernel = kernel[:,:,:,tf.newaxis, tf.newaxis] # image 3 dimension + No. of channel + No. of output channel

    output=tf.nn.conv3d(input, kernel, strides=(1,1,1,1,1), padding="SAME")

    return output

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""
     
    d = tf.distributions.Normal(mean, std)
     
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
     
    gauss_kernel = tf.einsum('i,j,k->ijk',
                                  vals,
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

if __name__=="__main__":

    from Dataset_Reader import Reader

    train, _, _ = Reader(r"../Training/File_reference.csv", r"../Data")

    test = list(train.take(1))[0]
    print(test.shape)

    print(_gaussianMatrix(3, 9))