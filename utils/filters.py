import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.signal import convolve2d as conv2d

@tf.function
def sobelFilter3D(input):
    """
    Description: this is an implementation of 3D sobel silter for tensorflow
    reference: 
    """
    smooth = np.array((1,2,1))
    derivative = np.array((1,0,-1))

    hx = smooth.reshape((1,3))
    hpx = derivative.reshape((1,3))

    hy = smooth.reshape((3,1))
    hpy = derivative.reshape((3,1))

    hz = smooth
    hpz = derivative

    Gx = np.zeros((3,3,3))
    Gy = np.zeros((3,3,3))
    Gz = np.zeros((3,3,3))

    for i in range(3):
        Gx[i,:,:] = hz[i] * conv2d(hpx, hy, mode="full") # h'_x = h'(x) :: h(y) :: h(z) | "::" stands for convolution; mode="full" is very important
        Gy[i,:,:] = hz[i] * conv2d(hx, hpy, mode="full") # h'_y = h'(y) :: h(x) :: h(z) | this is why we cannot use tf.nn.conv3d which only has 
        Gz[i,:,:] = hpz[i] * conv2d(hx, hy, mode="full") # h'_y = h'(z) :: h(x) :: h(y) | valid and same padding while scipy.signal has no conv3d. 

    Gx_tf = tf.reshape(tf.constant(Gx, dtype=tf.float32),(3,3,3,1,1)) # image 3 dimension + channel + output channel
    Gx_tf = tf.reshape(tf.constant(Gx, dtype=tf.float32),(3,3,3,1,1))
    Gx_tf = tf.reshape(tf.constant(Gx, dtype=tf.float32),(3,3,3,1,1))

    grad_x = tf.nn.conv3d(input, Gx_tf, strides=(1,)*5, padding="SAME")
    grad_y = tf.nn.conv3d(input, Gx_tf, strides=(1,)*5, padding="SAME")
    grad_z = tf.nn.conv3d(input, Gx_tf, strides=(1,)*5, padding="SAME")
    
    output = tf.math.sqrt(
        tf.math.square(grad_x)+tf.math.square(grad_y)+tf.math.square(grad_z))

    return output

@tf.function    
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

if __name__=="__main__":
    #test 1: zeros tensor input
    print("test 1: zeros tensor input")
    test = tf.zeros((1,6,6,6,1))
    gaussianFilter3D(test,0.1,3)

    #test 2: image input test
    print("test 2: image input test")
    import SimpleITK as sitk
    image = sitk.GetArrayFromImage(sitk.ReadImage(r"/uCTGan/data/unitTest/test_t1_brain.nii.gz"))
    print(image.shape)
    tfimage = tf.convert_to_tensor(image)    
    tfimage = tfimage[tf.newaxis,:,:,:,tf.newaxis]
    output = gaussianFilter3D(tfimage,0.5,3).numpy()
    output= np.squeeze(output)
    print(output.shape)
    sitk.WriteImage(sitk.GetImageFromArray(output), r"/uCTGan/data/unitTest/test_gaussian.nii.gz")

    output2 = sobelFilter3D(tfimage).numpy()
    output2 = np.squeeze(output2)
    sitk.WriteImage(sitk.GetImageFromArray(output2), r"/uCTGan/data/unitTest/test_sobel.nii.gz")
    #test 3: train step test