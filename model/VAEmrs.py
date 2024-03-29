# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)

import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense, Conv3DTranspose
from tensorflow.keras.layers import Input, Reshape, Flatten, Dropout, SpatialDropout3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from utils.group_norm import GroupNormalization
import tensorflow as tf
import numpy as np
import os

def green_block(inp, filters, data_format='channels_last', name=None):
    """
    green_block(inp, filters, name=None)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `inp` to the output. Used
    internally in the model. Can be used independently as well.
    Parameters
    ----------
    `inp`: An keras.layers.layer instance, required
        The keras layer just preceding the green block.
    `filters`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this many no. of channels.
    `data_format`: string, optional
        The format of the input data. Must be either 'chanels_first' or
        'channels_last'. Defaults to `channels_last`, as used in the paper.
    `name`: string, optional
        The name to be given to this green block. Defaults to None, in which
        case, keras uses generated names for the involved layers. If a string
        is provided, the names of individual layers are generated by attaching
        a relevant prefix from [GroupNorm_, Res_, Conv3D_, Relu_, ], followed
        by _1 or _2.
    Returns
    -------
    `out`: A keras.layers.Layer instance
        The output of the green block. Has no. of channels equal to `filters`.
        The size of the rest of the dimensions remains same as in `inp`.
    """
    inp_res = Conv3D(
        filters=filters,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    # axis=1 for channels_last data format
    # No. of groups = 8, as given in the paper
    x = GroupNormalization(
        groups=8,
        axis=-1 if data_format == 'channels_last' else 1,
        name=f'GroupNorm_1_{name}' if name else None)(inp)
    x = Activation('relu', name=f'Relu_1_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_1_{name}' if name else None)(x)

    x = GroupNormalization(
        groups=8,
        axis=-1 if data_format == 'channels_last' else 1,
        name=f'GroupNorm_2_{name}' if name else None)(x)
    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_2_{name}' if name else None)(x)

    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    return out


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_var) * epsilon


def dice_coefficient(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[1,2,3]) + 1e-8
    return K.mean(2 * intersection / dn, axis=[0, -1])


def loss_gt(e=1e-8):
    """
    loss_gt(e=1e-8)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper. This function
    only calculates - L<dice> term of the following equation. (i.e. GT Decoder part loss)
    
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    
    Parameters
    ----------
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
        
    Returns
    -------
    loss_gt_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss.
        
    """
    def loss_gt_(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[1,2,3]) + e
        
        return - K.mean(2 * intersection / dn, axis=[0,-1])
    
    return loss_gt_

def loss_VAE(input_shape, weight_L2=0.1, weight_KL=0.1):
    """
    loss_VAE(input_shape, z_mean, z_var, weight_L2=0.1, weight_KL=0.1)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper. This function
    calculates the following equation, except for -L<dice> term. (i.e. VAE decoder part loss)
    
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    
    Parameters
    ----------
     `input_shape`: A 4-tuple, required
        The shape of an image as the tuple (c, H, W, D), where c is
        the no. of channels; H, W and D is the height, width and depth of the
        input image, respectively.
    `z_mean`: An keras.layers.Layer instance, required
        The vector representing values of mean for the learned distribution
        in the VAE part. Used internally.
    `z_var`: An keras.layers.Layer instance, required
        The vector representing values of variance for the learned distribution
        in the VAE part. Used internally.
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
        
    Returns
    -------
    loss_VAE_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the L2 and KL loss.
        
    """
    def loss_VAE_(y_true, y_pred, z_mean, z_var,):
        c, H, W, D = input_shape
        n = c * H * W * D
        
        loss_L2 = K.mean(K.square(y_true - y_pred), axis=(1, 2, 3, 4)) # original axis value is (1,2,3,4).

        loss_KL = (1 / n) * K.sum(
            K.exp(z_var) + K.square(z_mean) - 1. - z_var,
            axis=-1
        )

        return K.mean(weight_L2 * loss_L2 + weight_KL * loss_KL)

    return loss_VAE_

def build_model(input_shape=(60, 60, 15, 6), output_channels=6):
    """
    build_model(input_shape=(4, 160, 192, 128), output_channels=3, weight_L2=0.1, weight_KL=0.1)
    -------------------------------------------
    Creates the model used in the BRATS2018 winning solution
    by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
    Parameters
    ----------
    `input_shape`: A 4-tuple, optional.
        Shape of the input image. Must be a 4D image of shape (c, H, W, D),
        where, each of H, W and D are divisible by 2^4, and c is divisible by 4.
        Defaults to the crop size used in the paper, i.e., (4, 160, 192, 128).
    `output_channels`: An integer, optional.
        The no. of channels in the output. Defaults to 3 (BraTS 2018 format).
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `dice_e`: Float, optional
        A small epsilon term to add in the denominator of dice loss to avoid dividing by
        zero and possible gradient explosion. This argument will be passed to loss_gt function.
    Returns
    -------
    `model`: A keras.models.Model instance
        The created model.
    """
    D, H, W, c = input_shape
    assert len(input_shape) == 4, "Input shape must be a 4-tuple"
    assert (c // 4) > 0, "The no. of channels must be more than 4"
    # assert (H % 16) == 0 and (W % 16) == 0 and (D % 16) == 0, \
    #     "All the input dimensions must be divisible by 16. ({}, {}, {}) is not a valid shape".format(D,H,W)


    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    ## Input Layer
    inp = Input(input_shape)

    ## The Initial Block
    x = Conv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_last',
        name='Input_x1')(inp)

    ## Dropout (0.2)
    x = SpatialDropout3D(0.2, data_format='channels_last')(x)

    ## Green Block x1 (output filters = 32)
    x1 = green_block(x, 32, name='x1')
    x = Conv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='valid',
        data_format='channels_last',
        name='Enc_DownSample_32')(x1)

    ## Green Block x2 (output filters = 64)
    x = green_block(x, 16, name='Enc_64_1')
    x2 = green_block(x, 16, name='x2')
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='valid',
        data_format='channels_last',
        name='Enc_DownSample_64')(x2)

    ## Green Blocks x2 (output filters = 128)
    x = green_block(x, 32, name='Enc_128_1')
    x3 = green_block(x, 32, name='x3')
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='valid',
        data_format='channels_last',
        name='Enc_DownSample_128')(x3)

    ## Green Blocks x4 (output filters = 256)
    x = green_block(x, 64, name='Enc_256_1')
    x4 = green_block(x, 128, name='Enc_256_2')
    x = green_block(x4, 128, name='Enc_256_3')
    x = green_block(x, 64, name='x4')
    ### Output Block
    out_GT = Conv3D(
        filters=output_channels,  # No. of tumor classes is 3
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        activation='relu',
        name='Dec_GT_Output')(x)
    '''
    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    ## GT (Groud Truth) Part
    # -------------------------------------------------------------------------

    ### Green Block x1 (output filters=128)
    x = Conv3D(
        filters=128,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_GT_ReduceDepth_128')(x4)
    x = UpSampling3D(
        size=2,
        data_format='channels_last',
        name='Dec_GT_UpSample_128')(x)
    x = Add(name='Input_Dec_GT_128')([x, x3])
    x = green_block(x, 128, name='Dec_GT_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_GT_ReduceDepth_64')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_last',
        name='Dec_GT_UpSample_64')(x)
    x = Add(name='Input_Dec_GT_64')([x, x2])
    x = green_block(x, 64, name='Dec_GT_64')

    ### Green Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_GT_ReduceDepth_32')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_last',
        name='Dec_GT_UpSample_32')(x)
    x = Add(name='Input_Dec_GT_32')([x, x1])
    x = green_block(x, 32, name='Dec_GT_32')
    ### Blue Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_last',
        name='Input_Dec_GT_Output')(x)

    ### Output Block
    out_GT = Conv3D(
        filters=output_channels,  # No. of tumor classes is 3
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        activation='sigmoid',
        name='Dec_GT_Output')(x)
    '''
    ## VAE (Variational Auto Encoder) Part
    # -------------------------------------------------------------------------

    ### VD Block (Reducing dimensionality of the data)
    x = GroupNormalization(groups=8, axis=-1, name='Dec_VAE_VD_GN')(x4)
    x = Activation('relu', name='Dec_VAE_VD_relu')(x)
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_last',
        name='Dec_VAE_VD_Conv3D')(x)

    # Not mentioned in the paper, but the author used a Flattening layer here.
    x = Flatten(name='Dec_VAE_VD_Flatten')(x)
    x = Dense(256, name='Dec_VAE_VD_Dense')(x)

    ### VDraw Block (Sampling)
    z_mean = Dense(128, name='Dec_VAE_VDraw_Mean')(x)
    z_var = Dense(128, name='Dec_VAE_VDraw_Var')(x)
    x = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, z_var])
    
    ### VU Block (Upsizing back to a depth of 256)
    x = Dense(7*7*8)(x)
    x = Activation('relu')(x)
    x = Reshape((7,7,1,8))(x)
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_VAE_ReduceDepth_256')(x)

    x = Conv3DTranspose(
        filters=64,
        kernel_size=(3,3,3),
        strides=(2,2,2),
        data_format="channels_last",
        name='Dec_VAE_Conv3Dtranpose64')(x)

    ### Green Block x1 (output filters=128)
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_VAE_ReduceDepth_32')(x)
    x = Conv3DTranspose(
        filters=32,
        kernel_size=(2,2,3),
        strides=(2,2,2),
        data_format="channels_last",
        name='Dec_VAE_Conv3Dtranpose32')(x)
    x = green_block(x, 32, name='Dec_VAE_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(
        filters=16,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_VAE_ReduceDepth_64')(x)
    x = Conv3DTranspose(
        filters=16,
        kernel_size=(2,2,3),
        strides=(2,2,2),
        data_format="channels_last",
        name='Dec_VAE_Conv3Dtranpose16')(x)
    x = green_block(x, 16, name='Dec_VAE_64')

    ### Green Block x1 (output filters=32)
    # x = Conv3D(
    #     filters=8,
    #     kernel_size=(1, 1, 1),
    #     strides=1,
    #     data_format='channels_last',
    #     name='Dec_VAE_ReduceDepth_8')(x)
    # x = UpSampling3D(
    #     size=2,
    #     data_format='channels_last',
    #     name='Dec_VAE_UpSample_8')(x)
    # x = green_block(x, 8, name='Dec_VAE_32')

    ### Blue Block x1 (output filters=32)
    x = Conv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_last',
        name='Input_Dec_VAE_Output')(x)

    ### Output Block
    out_VAE = Conv3D(
        filters=c,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='Dec_VAE_Output')(x) 

    # Build and Compile the model
    out = out_GT
    model = Model(inp, outputs=[out, out_VAE, z_mean, z_var])  # Create the model

    return model

class VAEmrs:
    def __init__(self, 
            input_shape, 
            output_channels, 
            weight_L2=0.1, 
            weight_KL=0.1, 
            lr1=0.0001, 
            lr2=0.0001):
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.vae_loss_func = loss_VAE(input_shape, weight_L2=weight_L2, weight_KL=weight_KL)
        self.mrs_loss_func = MeanSquaredError()
        self.optimizer1 = Adam(lr1)
        self.optimizer2 = Adam(lr2)
        self.model = build_model(
            input_shape=input_shape, 
            output_channels=output_channels
        )
    def fetch_batch(self, trainset, valset):
        for trainbatch, valbatch in zip(trainset, valset):
            train_x = trainbatch[0]
            train_mrs = trainbatch[1]
            val_x = valbatch[0]
            val_mrs = valbatch[1]
            yield train_x, train_mrs, val_x, val_mrs
    
    @tf.function
    def __train_step__(self, train_x, train_mrs, val_x, val_mrs, validate=True):
        with tf.GradientTape() as vae_tape, tf.GradientTape() as mrs_tape:
            mrs, x_pred, z_mean, z_var = self.model(train_x)
            vae_loss = self.vae_loss_func(train_x, x_pred, z_mean, z_var)
            mrs_loss = self.mrs_loss_func(train_mrs, mrs) *0.01
        
        gradients_vae = vae_tape.gradient(vae_loss, self.model.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients_vae, self.model.trainable_variables))
        gradients_mrs = mrs_tape.gradient(mrs_loss, self.model.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients_mrs, self.model.trainable_variables))

        if validate:
            val_mrs_pred, val_x_pred, val_z_mean, val_z_var = self.model(val_x)
            val_vae_loss = self.vae_loss_func(val_x, val_x_pred, val_z_mean, val_z_var)
            val_mrs_loss = self.mrs_loss_func(val_mrs, val_mrs_pred)
            return [mrs_loss, vae_loss, val_mrs_loss, val_vae_loss], [val_x, val_mrs, val_x_pred, val_mrs_pred]
        else:
            return [mrs_loss, vae_loss]

    def train_step(self, datapipeline, batch_size=None, validate=True):
        train_x, train_mrs, val_x, val_mrs= next(datapipeline)
        if validate:
            [mrs_loss, vae_loss, val_mrs_loss, val_vae_loss], val_output = self.__train_step__(train_x, train_mrs, val_x, val_mrs, validate=validate)
            history = {
                "mrs_loss":mrs_loss,
                "vae_loss":vae_loss,
                "val_mrs_loss": val_mrs_loss,
                "val_vae_loss": val_vae_loss
            }
            return history, val_output
        else:
            mrs_loss, vae_loss = self.__train_step__(train_x, train_mrs, val_x, val_mrs, validate=validate)
            history = {
                "mrs_loss":mrs_loss,
                "vae_loss":vae_loss,
            }
            return history
    def train(self, train_set, val_set, n_epochs, batch_size=None, logdir=r"data/Gan_training/log", logstart=500, logimage=8, logslices=slice(None)):
        summary_writer = tf.summary.create_file_writer(logdir)
        summary_writer.set_as_default()
        datapipeline=self.fetch_batch(train_set, val_set)
        for epoch in range(1,n_epochs):
            history, val_output = self.train_step(datapipeline, batch_size=batch_size, validate=True)
            with summary_writer.as_default():
                for k in history.keys():
                    history[k] = np.squeeze(history[k].numpy())
                    tf.summary.scalar(k, data=history[k], step=epoch)
            if epoch == 1:
                loss_min = history["val_mrs_loss"]
                loss_min_epoch = 1
                summary = {k:[] for k in history.keys()}
            else:
                for key in summary.keys():
                    summary[key].append(history[key])
            
            if history["mrs_loss"] < loss_min:
                loss_min = history["val_mrs_loss"]
                loss_min_epoch = epoch
                if epoch > logstart:
                    self.model.save(os.path.join(logdir, "model_epoch_{}.h5".format(epoch)))
                    if logimage:
                        self.save_image(val_output, epoch, logdir, logimage, logslices=logslices)

            print("Epoch -- {} -- CurrentBest -- {} -- val_mrs_loss -- {:.4f}".format(epoch, loss_min_epoch, loss_min))
            print("   ".join(["{}: {:.4f}".format(k, v) for k, v in history.items()]))
        return summary

    def save_image(self, output, epoch, logdir=".", logimage=8, logslices=slice(None)):
        D,H,W,C = self.input_shape
        mrs_c = self.output_channels

        def getArray(i):
            if isinstance(i, tf.Tensor):
                return np.squeeze(i.numpy())
            elif isinstance(i, np.ndarray):
                return np.squeeze(i)
            else:
                raise ValueError("Ether a tf.Tensor or np.ndarray is expected")

        # def concatChannel(image):
        #     '''
        #     image shape = (y, x, channel)
        #     '''
        #     return np.concat([image[:,:,i] for i in image.shape[-1]], axis = )

        val_x, val_mrs, val_x_pred, val_mrs_pred = [getArray(i)[logslices] for i in output]
        val_x.dump(os.path.join(logdir, "val-x-{}.png".format(epoch)))
        val_mrs.dump(os.path.join(logdir, "val-mrs-{}.png".format(epoch)))
        val_x_pred.dump(os.path.join(logdir, "val-x-pred-{}.png".format(epoch)))
        val_mrs_pred.dump(os.path.join(logdir, "val-mrs-pred-{}.png".format(epoch)))


        # image = np.squeeze(output[0].numpy())[:logimage]
        # valimage = np.squeeze(output[1])[:logimage]

        # if logslices:
        #     image = image[logslices]
        #     valimage = valimage[logslices]
        #     shape = image.shape
        # else:
        #     shape = image.shape
        #     image = image[:, shape[1]//2, ...]
        #     valimage = valimage[:, shape[1]//2, ...]

        # image = np.concatenate([image.reshape((-1, shape[-1])), valimage.reshape((-1, shape[-1]))], axis=1) * 255

        # # fig = plt.figure()
        # # plt.imshow(image, cmap="gray")
        # # plt.axis("off")
        # # plt.savefig(os.path.join(logdir, "val-image-{}.png".format(epoch)), dpi=300)
        # # plt.close()
        # imsave(os.path.join(logdir, "val-image-{}.png".format(epoch)), image.astype(np.uint8), check_contrast=False)

    def save_output(self, checkpount):
        pass
    def save_model(self, ):
        pass
    def load_model(self, ):
        pass
