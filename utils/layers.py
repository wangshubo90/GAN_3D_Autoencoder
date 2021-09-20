
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as bk
from tensorflow.keras.layers import Conv3DTranspose, Conv3D, BatchNormalization, Activation, Add, LSTM, Lambda, concatenate
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

def residual_block(input, filters = 64, kernel_size= 3, strides = (1,1,1), padding = "SAME", activate=relu, **kwargs):
    identity = input
    
    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=strides, padding = padding, **kwargs)(input)
    x = BatchNormalization()(x)
    x = activate(x)

    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=(1,1,1), padding = padding, **kwargs)(x)
    x = BatchNormalization()(x)    

    if np.prod(strides) > 1:
        identity = Conv3D(filters=filters, kernel_size=1, strides=strides, padding = padding, **kwargs)(identity)
        identity = BatchNormalization()(identity)  
    else:
        pass
    
    x = x + identity
    x = activate(x)
    
    return x

def resBN_block(input, filters = 64, kernel_size= 3, strides = (1,1,1),  padding = "SAME", compression=2, activate=relu, **kwargs):
    identity = input
    
    x = Conv3DTranspose(filters = filters // compression, kernel_size=1, strides=strides, padding = padding, **kwargs)(input)
    x = BatchNormalization()(x)
    x = activate(x)
    
    x = Conv3D(filters = filters // compression, kernel_size=kernel_size, strides=(1,1,1), padding = padding, **kwargs)(x)
    x = BatchNormalization()(x)
    x = activate(x)
    
    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=(1,1,1), padding = padding, **kwargs)(x)
    x = BatchNormalization()(x)

    if np.prod(strides) > 1:
        identity = Conv3D(filters=filters, kernel_size=1, strides=strides, padding = padding, **kwargs)(identity)
        identity = BatchNormalization()(identity)    
    else:
        pass
    
    x = x + identity
    x = activate(x)
    
    return x

def resTP_block(input, filters = 64, kernel_size= 3, strides = (1,1,1), padding = "SAME", activate=relu, **kwargs):
    identity = input
    
    x = Conv3DTranspose(filters = filters, kernel_size=kernel_size, strides=strides, padding = padding, **kwargs)(input)
    x = BatchNormalization()(x)
    x = activate(x)

    x = Conv3D(filters = filters, kernel_size=kernel_size, strides=(1,1,1), padding = padding, **kwargs)(x)
    x = BatchNormalization()(x)    

    if np.prod(strides) > 1:
        identity = Conv3DTranspose(filters=filters, kernel_size=1, strides=strides, padding = padding, **kwargs)(identity)
        identity = BatchNormalization()(identity)  
    else:
        pass
    
    x = x + identity
    x = activate(x)

    return x

class GlobalSumPooling3D():
    def __init__(self, kernel_size):
        pass
        
    def __call__(self):
        pass

def spatialLSTM3D(input, mask, lstm_activation='tanh'):
    '''
        input shape: [batch, seq_step, depth, height, width, channel]
    '''
    org_shape = bk.shape(input)
    input = bk.reshape(input, shape=(org_shape[0], org_shape[1], org_shape[2]*org_shape[3]*org_shape[4], org_shape[-1])) # shape [batch, step, 3dflatten, channel]
    new_shape = bk.shape(input)
    lstmlayer1 = LSTM(org_shape[-1], activation=lstm_activation, return_sequences=True)
    lstmlayer2 = LSTM(org_shape[-1], activation=lstm_activation, return_sequences=True)
    lstmlayer3 = LSTM(org_shape[-1])

    spatialpath = []
    for i in range(new_shape[2]):
        x = Lambda(lambda z: z[:,:,i,:])(input)
        x = lstmlayer1(x, mask=mask)
        x = lstmlayer2(x, mask=mask)
        x = lstmlayer3(x, mask=mask)   # x shape : [batch, channel]
        x = tf.expand_dims(x, axis=1)
        spatialpath.append(x)
    x = concatenate(spatialpath, axis=1)
    x = bk.reshape(x, shape=(org_shape[0], *org_shape[2:]))
    return x
    

if __name__=="__main__":

    input = tf.zeros(shape = (3, 96, 100, 100, 1))
    y1 = residual_block(input, strides=(2,2,2))
    y2 = resBN_block(input, strides=(2,2,2))