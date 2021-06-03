
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import conv3d, relu
from tensorflow.keras.layers import Conv1D, Conv3D, BatchNormalization, Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.activations import relu
from tensorflow.python.keras.layers.convolutional import Conv1DTranspose

def __default_conv3D(input, filters=8, kernel_size=3, strides=(1,1,1), weight_decay = 1e-4, **kwargs):
    '''
    Description: set up defaut parameters for Conv3D layers
    '''
    DefaultConv3D = partial(
        Conv3D, 
        filters = filters,
        kernel_size=kernel_size, 
        strides=strides,
        padding="SAME", 
        use_bias=True, 
        kernel_regularizer = keras.regularizers.l2(weight_decay),
        kernel_initializer="he_normal",
        **kwargs
    )
    return DefaultConv3D()(input)

def __bottleneck_layer(input, filters = 64, kernel_size = 3, strides = (1,1,1), cardinality = 16, weight_decay = 5e-4):
    '''
    Description: bottleneck layer for a single path(cardinality = 1)
    Args:   input: input tensor
            filters : number of filters for the last layer in a single path, suppose to be total number
                        of filters // cardinality of ResNeXt block.
            strides : strides, must be tuple of 3 elements
    '''
    x = input

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = 1, strides = strides, weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = kernel_size, strides = (1,1,1), weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters, kernel_size = 1, strides = (1,1,1), weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def resnext_block(input, filters = 64, kernel_size = 3, strides = (1,1,1), cardinality = 16, weight_decay = 5e-4):
    '''
    Description: refer to the ResNeXt architechture. One ResNeXt_block contains several paths (cardinality) of bottleneck layers joint by a skip connection.
    '''

    if strides[0] == 1:
        init = input
    elif strides[0] > 1:
        init = __default_conv3D(input, filters = filters, kernel_size=kernel_size, strides=strides, weight_decay = weight_decay)
        init = BatchNormalization()(init)

    x = [init]
    
    for i in range(cardinality):
        x_sub = __bottleneck_layer(input, filters = filters, kernel_size=kernel_size, strides=strides, cardinality=cardinality, weight_decay=weight_decay)
        x_sub = BatchNormalization()(x_sub)
        x.append(x_sub)

    x = Add()(x)
    x = Activation('relu')(x)

    return x

def residual_block(input, filters = 64, kernel_size= 3, strides = (1,1,1), **kwargs):
    identity = input
    
    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=strides, **kwargs)(input)
    x = BatchNormalization()(x)
    x = relu(x)
    
    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=(1,1,1), **kwargs)(x)
    
    if np.prod(strides) > 1:
        identity = Conv3D(filters=filters, kernel_size=1, strides=strides, **kwargs)(identity)
    else:
        pass
    
    x = x + identity
    
    x = BatchNormalization()(x)
    x = relu(x)
    
    return x

def resBN_block(input, filters = 64, kernel_size= 3, strides = (1,1,1), compression=2, **kwargs):
    identity = input
    
    x = Conv3D(filters = filters // compression, kernel_size=1, strides=strides, **kwargs)(input)
    x = BatchNormalization()(x)
    x = relu(x)
    
    x = Conv3D(filters = filters // compression, kernel_size=kernel_size, strides=(1,1,1), **kwargs)(x)
    x = BatchNormalization()(x)
    x = relu(x)
    
    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=(1,1,1), **kwargs)(x)

    if np.prod(strides) > 1:
        identity = Conv3D(filters=filters, kernel_size=1, strides=strides, **kwargs)(identity)
    else:
        pass
    
    x = x + identity
    
    x = BatchNormalization()(x)
    x = relu(x)
    
    return x
