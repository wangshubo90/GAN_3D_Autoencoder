import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.stats as st


def gaussianFilter3D(input, sigma, kernel_size=None):
    """
    apply 3d guassian filter on a 3d image (a 5D tf.tensor)
    """
    if kernel_size == None:
        kernel_size = int(sigma * 6)+1
    
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy, zz= np.meshgrid(ax, ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.power(sigma,2))
    kernel = kernel / np.sum(kernel)

    kernel = tf.constant(kernel, dtype=tf.float32)
    kernel = kernel[:,:,:,tf.newaxis, tf.newaxis] # image 3 dimension + No. of channel + No. of output channel

    output=tf.nn.conv3d(input, kernel, strides=(1,1,1,1,1), padding="SAME")

    return output

# def gaussian_kernel(size: int,
#                     mean: float,
#                     std: float,
#                    ):
#     """Makes 2D gaussian Kernel for convolution."""
     
#     d = tf.distributions.Normal(mean, std)
     
#     vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
     
#     gauss_kernel = tf.einsum('i,j,k->ijk',
#                                   vals,
#                                   vals,
#                                   vals)

#     return gauss_kernel / tf.reduce_sum(gauss_kernel)

if __name__=="__main__":
    #test 1: zeros tensor input
    print("test 1: zeros tensor input")
    test = tf.zeros((1,6,6,6,1))
    gaussianFilter3D(test,0.1,3)

    #test 2: image input test
    print("test 2: image input test")
    import SimpleITK as sitk
    image = sitk.GetArrayFrom(sitk.ReadImage(r"/uCTGan/data/unitTest/test_t1_brain.nii.gz"))
    print(image.shape)
    tfimage = tf.convert_to_tensor(image)    
    tfimage = tfimage[tf.newaxis,:,:,:,tf.newaxis]
    output = gaussianFilter3D(tfimage,0.5,3).numpy()
    output= np.squeeze(output)
    print(output.shape)
    sitk.WriteImage(sitk.GetImageFromArray(output), r"/uCTGan/data/unitTest/test_gaussian.nii.gz")

    #test 3: train step test
    