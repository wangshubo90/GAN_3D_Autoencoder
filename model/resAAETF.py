#!/home/spl/ml/sitk/bin/python
import os
import numpy as np
from random import sample, seed
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv1D, Conv3DTranspose, Flatten, Reshape, Input, BatchNormalization
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from utils.layers import *
import matplotlib.pyplot as plt

class resAAE():
    #Adversarial Autoencoder
    def __init__(self, 
                img_shape=(48, 96, 96, 1), 
                encoded_dim=16, 
                loss_AE = "mse", 
                loss_GD = "mse",
                acc = "mse",
                g_loss_factor = .1,
                hidden = (32,64,128,256),
                hidden_D = (32,64,128),
                output_slices=slice(None),
                last_encoder_act=relu,
                last_decoder_act=sigmoid,
                d_dropout=0.8,
                **kwargs):
        self.encoded_dim = encoded_dim
        self.hidden=hidden
        self.hidden_D=hidden_D
        self.loss_function_AE = loss_AE
        self.loss_function_GD = loss_GD
        self.acc_function = acc
        self.output_slices=output_slices
        self.gfactor=g_loss_factor
        self.d_dropout=d_dropout
        self.optimizer_discriminator = Adam(kwargs["optD_lr"], beta_1=tf.Variable(kwargs["optD_beta"]))
        self.optimizer_autoencoder = Adam(kwargs["optAE_lr"], beta_1=tf.Variable(kwargs["optAE_beta"]))
        self.img_shape = img_shape
        self.last_encoder_act=last_encoder_act
        self.last_decoder_act=last_decoder_act
        self.initializer = RandomNormal(mean=0., stddev=1.)
        self.encoder, self.decoder, self.autoencoder, self.discriminator = self._modelBuild(
                self.img_shape
                )

    def _buildEncoder(self, input_shape, filters=[16, 32, 64, 128], last_activation=relu):
        
        input = Input(shape=input_shape)
        x = Conv3D(filters=filters[0], kernel_size=5, strides=(2,2,2), padding="SAME")(input)
        x = BatchNormalization()(x)
        x = relu(x)
        for i, ft in enumerate(filters[1:]):
            if i == len(filters[1:])-1:
                x = residual_block(x, filters = ft, kernel_size= 3,  
                            strides = (2,2,2), padding = "SAME", activate=relu)
            else:
                x = residual_block(x, filters = ft, kernel_size= 3,  
                            strides = (2,2,2), padding = "SAME", activate=last_activation)
        
        encoder = Model(inputs=input, outputs=x)        
        return encoder
     
    def _buildDecoder(self, input_shape, filters=[16, 32, 64, 128], last_activation=relu, **kwargs):
        input = Input(shape=input_shape)
        x = input
        for i, ft in enumerate(filters[-1:0:-1]):
            if i != len(filters[-2::-1])-1:
                x = resTP_block(x, filters=ft, strides=(2,2,2),padding="SAME")
            else:
                x = resTP_block(x, filters=ft, strides=(2,2,2),padding="SAME", activation="relu")
        
        x = Conv3DTranspose(filters=filters[0], kernel_size=3, strides=(2,)*3, padding="SAME", activation="relu")(x)
        x = BatchNormalization()(x)

        if "slices" in kwargs:
            slices = kwargs["slices"]
            x = x[slices]

        x = Conv3DTranspose(filters=1, kernel_size=3, strides=(1,)*3, padding="SAME", activation=last_activation)(x)
        decoder = Model(inputs=input, outputs=x)
        return decoder

    def _buildDiscriminator(self, input_shape, filters=[16, 32, 64, 128], last_activation=relu):

        input = Input(shape=input_shape)
        x = Conv3D(filters=filters[0], kernel_size=5, strides=(2,2,2), padding="SAME")(input)
        x = BatchNormalization()(x)
        x = relu(x)
        for i, ft in enumerate(filters[1:]):
            if i == len(filters[1:])-1:
                x = residual_block(x, filters = ft, kernel_size= 3,  
                            strides = (2,2,2), padding = "SAME", activate=relu)
            else:
                x = residual_block(x, filters = ft, kernel_size= 3,  
                            strides = (2,2,2), padding = "SAME", activate=last_activation)
        
        x = GlobalAveragePooling3D()(x)
        x = Flatten()(x)
        x = Dropout(self.d_dropout)(x)
        x = Dense(filters[-1]//2)(x)
        x = Dropout(self.d_dropout)(x)
        x = Dense(filters[-1]//2)(x)
        x = Dropout(self.d_dropout)(x)
        x = Dense(1, activation=last_activation)(x)
        discriminator = Model(inputs=input, outputs=x) 

        return discriminator

    def _modelBuild(self, input_shape):

        encoder=self._buildEncoder(input_shape, filters=self.hidden, last_activation=self.last_encoder_act)
        decoder=self._buildDecoder(encoder.output_shape[1:], filters=self.hidden, last_activation=self.last_decoder_act, slices=self.output_slices)
        
        autoencoder_input = Input(shape = input_shape)
        autoencoder=Model(autoencoder_input, decoder(encoder(autoencoder_input)))
        discriminator=self._buildDiscriminator(input_shape, filters=self.hidden_D, last_activation=self.last_decoder_act)
        assert autoencoder.output_shape == autoencoder.input_shape , "shape incompatible for autoencoder"
        #autoencoder.compile(optimizer=optimizer_autoencoder, loss=self.loss_function_AE, metrics=self.acc_function)
        # discriminator.trainable = False
        # generator = Model(autoencoder_input, discriminator(decoder(encoder(autoencoder_input))))
        # generator.compile(optimizer=optimizer_generator, loss=self.loss_function_GD)
        # discriminator.trainable = True
        # discriminator.compile(optimizer=optimizer_discriminator, loss=self.loss_function_GD)
    

        return encoder, decoder, autoencoder, discriminator

    @tf.function
    def __train_step__(self, x, x2, val_x, gfactor, validate=True):
        with tf.GradientTape() as ae_tape:
            fake_image = self.autoencoder(x, training = True)
            fake_output = self.discriminator(fake_image, training=False)

            ae_loss = self.loss_function_AE(x, fake_image)
            g_loss = self.loss_function_GD(fake_output, tf.ones_like(fake_output))
            total_loss = ae_loss + g_loss*gfactor
            ae_acc = self.acc_function(x, fake_image)
        
        ae_grad = ae_tape.gradient(total_loss, self.autoencoder.trainable_variables)
        self.optimizer_autoencoder.apply_gradients(zip(ae_grad, self.autoencoder.trainable_variables))
        
        with tf.GradientTape() as disc_tape:
            fake_image = self.autoencoder(x, training = False)
            fake_output = self.discriminator(fake_image, training=True)
            real_output = self.discriminator(x2, training=True)

            d_label = tf.concat([tf.zeros_like(fake_output), tf.ones_like(real_output)], axis=0)
            d_loss = self.loss_function_GD(tf.concat([fake_output, real_output], axis=0), d_label)

        d_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        if validate:
            val_output = self.autoencoder(val_x, training=False)
            val_loss = self.loss_function_AE(val_x, val_output)
            val_acc = self.acc_function(val_x, val_output)
            return [ae_loss, ae_acc, d_loss, g_loss, val_loss, val_acc, total_loss], val_output
        else:
            return ae_loss, ae_acc, d_loss, g_loss, total_loss

    def train_step(self, train_set, val_set, batch_size, validate=True):
        x_idx_list = sample(range(train_set.shape[0]), batch_size)
        x_idx_list2 = sample(range(train_set.shape[0]), batch_size)
        x = train_set[x_idx_list]
        x2 = train_set[x_idx_list2]
        val_x = val_set[sample(range(val_set.shape[0]), batch_size*2)]
        if validate:
            [ae_loss, ae_acc, d_loss, g_loss, val_loss, val_acc, total_loss], val_output= self.__train_step__(x, x2, val_x, self.gfactor, validate=validate)
            
            history = {
                        'AE_loss':ae_loss,
                        'total_loss': total_loss,
                        'AE_acc':ae_acc,
                        'D_loss':d_loss, 
                        'G_loss':g_loss,
                        'val_loss':val_loss,
                        "val_acc":val_acc
                    }
            
            return history, val_output
        else:
            ae_loss, ae_acc, d_loss, g_loss, total_loss = self.__train_step__(x, x2, val_x, self.gfactor, validate=validate)
            
            history = {
                        'AE_loss':ae_loss,
                        'total_loss': total_loss,
                        'AE_acc':ae_acc,
                        'D_loss':d_loss, 
                        'G_loss':g_loss,
                    }
            
            return history

    def train(self, train_set, val_set, batch_size, n_epochs, logdir=r"data/Gan_training/log", logstart=500, logimage=8):

        for epoch in np.arange(1, n_epochs):
            history, val_output = self.train_step(train_set, val_set, batch_size)

            if epoch == 1:
                loss_min = history["val_loss"]
                loss_min_epoch = 1
                summary = {k:[] for k in history.keys()}
            
            if epoch > logstart and history["val_loss"] < loss_min:
                loss_min = min(history["val_loss"], loss_min)
                loss_min_epoch = epoch
                self.autoencoder.save(os.path.join(logdir, "autoencoder_epoch_{}.h5".format(epoch)))
                if logimage:
                    self.save_image(val_output, epoch, logdir, logimage)
                #self.discriminator.save("../GAN_log/discriminator_epoch_{}.h5".format(epoch))
            
            print("Epoch -- {} -- CurrentBest -- {} -- val-loss -- {:.4f}".format(epoch, loss_min_epoch, loss_min))
            
            print("   ".join(["{}: {:.4f}".format(k, v) for k, v in history.items()])
            )
        
        return summary
    
    def save_image(output, epoch, logdir=".", logimage=8):
        image = np.squeeze(output.numpy())[:logimage]
        shape = image.shape
        image = image[:, shape[1]//2, ...]
        mid = ( shape[0] + 1 ) // 2 
        image = np.concatenate([image[:mid].reshape((-1, shape[-1])), image[mid:].reshape((-1, shape[-1]))], axis=1)

        fig = plt.figure()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "val-image-{}.png".format(epoch)), dpi=300)
        plt.close()

    def load_model(self):

        return

    def encodeImage(self):

        return 

    def plot_losses(self):

        return

if __name__ == "__main__":
    from tqdm import tqdm
    

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

    config={
        "optG_lr":0.01,
        "optG_beta":0.01,
        "optD_lr":0.01,
        "optD_beta":0.01,
        "optAE_lr":0.01,
        "optAE_beta":0.01,
        "img_shape": (15, 60, 60, 1), 
        "encoded_dim": 16, 
        "loss": "mse", 
        "acc": "mse",
        "hidden": (16, 32, 64, 128),
        "output_slices": [slice(None), slice(None,15), slice(2,62), slice(2,62), slice(None)],
        "batch_size": 16,
        "epochs": 5000
    }

    model = resAAE(**config)
    
    # import os
    # import SimpleITK as sitk 

    # datapath = r'../Data'
    # file_reference = r'../training/File_reference.csv'

    # img_ls = os.listdir(datapath)
    # train_set = np.zeros(shape=[len(img_ls), 48, 96, 96, 1])

    # idx = 0
    # for file in tqdm(img_ls):
    #     img = sitk.ReadImage(os.path.join(datapath, file))
    #     img = sitk.GetArrayFromImage(img)
    #     img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
    #     train_set[idx] = img
    #     idx += 1

    # model = AAE(encoded_dim=128)

    # batch_size=12
    # n_epochs=3000
    # seed=42
    # np.random.seed(42)

    # history = model.train(train_set, batch_size, n_epochs, len(img_ls))

    # import matplotlib as plt

    # fig, ax = plt.subplots(2, 2, figsize=(16,12))

    # ax[0].plot(range(2999), history["AE_loss"], label="AE_loss")
    # ax[0].title("AE_loss")

    # fig, ax = plt.subplots(2,2)

    # ax[0, 0].plot(range(2999), history["AE_loss"], label="AE_loss")
    # ax[0, 0].set_title("AE_loss")

    # ax[0, 1].plot(range(2999), history["G_loss"], label="G_loss")
    # ax[0, 1].set_title("G_loss")

    # ax[1, 0].plot(range(2999), history["D_loss"], label="D_loss")
    # ax[1, 0].set_title("D_loss")

    # plt.show()