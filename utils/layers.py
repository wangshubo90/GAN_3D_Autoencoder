
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
    
    x = tf.keras.layers.Add()([x , identity])
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
    
    x = tf.keras.layers.Add()([x , identity])
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
    
    x = tf.keras.layers.Add()([x , identity])
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
    
class VASampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, **kwargs):
        super(VASampling, self).__init__(**kwargs)
        self.n_batch = ""
        self.n_dim = ""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(VASampling, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))

class SimpleDense(tf.keras.layers.Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, adj, dropout, sparse=False, feature_nnz=0, act=tf.nn.relu, name=None):
        super(GCNLayer, self).__init__(name)
        self.adj = adj
        self.dropout = dropout
        self.sparse = sparse
        self.feature_nnz = feature_nnz
        self.act = act
        with tf.variable_scope(self.name):
            self.weights = glorot([input_dim, output_dim], name='weight')
            self.vars = [self.weights]

    def build(self, input_shape):
        self.w = 

    def call(self, inputs):
        x = inputs
        x = sparse_dropout(x, 1 - self.dropout, self.feature_nnz) if self.sparse else tf.nn.dropout(x, 1 - self.dropout)
        x = dot(x, self.weights, sparse=self.sparse)
        x = dot(self.adj, x, sparse=True)
        return self.act(x)


if __name__=="__main__":

    input = tf.zeros(shape = (3, 96, 100, 100, 1))
    y1 = residual_block(input, strides=(2,2,2))
    y2 = resBN_block(input, strides=(2,2,2))