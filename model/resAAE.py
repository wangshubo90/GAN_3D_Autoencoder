#!/home/spl/ml/sitk/bin/python
import os
import numpy as np
from random import sample, seed
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
from tqdm import tqdm
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv1D, Conv3DTranspose, Flatten, Reshape, Input, BatchNormalization
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from utils.layers import *

class resAAE():
    #Adversarial Autoencoder
    def __init__(self, 
                img_shape=(48, 96, 96, 1), 
                encoded_dim=16, 
                loss_AE = "mse", 
                loss_GD = "mse",
                acc = "mse",
                hidden = (32,64,128,256),
                output_slices=slice(None),
                last_encoder_act=relu,
                last_decoder_act=sigmoid,
                **kwargs):
        self.encoded_dim = encoded_dim
        self.hidden=hidden
        self.loss_function_AE = loss_AE
        self.loss_function_GD = loss_GD
        self.acc_function = acc
        self.output_slices=output_slices
        self.optimizer_generator = Adam(kwargs["optG_lr"], beta_1=tf.Variable(kwargs["optG_beta"]), clipnorm=1.0)
        self.optimizer_discriminator = Adam(kwargs["optD_lr"], beta_1=tf.Variable(kwargs["optD_beta"]))
        self.optimizer_autoencoder = Adam(kwargs["optAE_lr"], beta_1=tf.Variable(kwargs["optAE_beta"]))
        self.img_shape = img_shape
        self.last_encoder_act=last_encoder_act
        self.last_decoder_act=last_decoder_act
        self.initializer = RandomNormal(mean=0., stddev=1.)
        self.encoder, self.decoder, self.autoencoder, self.discriminator, self.generator = self._modelCompile(
                self.img_shape, self.encoded_dim, \
                self.optimizer_autoencoder,\
                self.optimizer_discriminator,\
                self.optimizer_generator
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
        x = Dense(128)(x)
        x = Dropout(0.7)(x)
        x = Dense(128)(x)
        x = Dropout(0.7)(x)
        x = Dense(1, activation=last_activation)(x)
        discriminator = Model(inputs=input, outputs=x) 

        return discriminator

    def _modelCompile(self, input_shape, encoded_dim, optimizer_autoencoder, optimizer_discriminator, optimizer_generator):

        encoder=self._buildEncoder(input_shape, filters=self.hidden, last_activation=self.last_encoder_act)
        decoder=self._buildDecoder(encoder.output_shape[1:], filters=self.hidden, last_activation=self.last_decoder_act, slices=self.output_slices)
        
        autoencoder_input = Input(shape = input_shape)
        autoencoder=Model(autoencoder_input, decoder(encoder(autoencoder_input)))
        assert autoencoder.input_shape == autoencoder.output_shape
        autoencoder.compile(optimizer=optimizer_autoencoder, loss=self.loss_function_AE, metrics=self.acc_function)
        #assert autoencoder.output_shape == autoencoder.input_shape , "shape incompatible for autoencoder"
        discriminator=self._buildDiscriminator(input_shape, filters=self.hidden, last_activation=self.last_decoder_act)
        discriminator.trainable = False
        generator = Model(autoencoder_input, discriminator(decoder(encoder(autoencoder_input))))
        generator.compile(optimizer=optimizer_generator, loss=self.loss_function_GD)
        discriminator.trainable = True
        discriminator.compile(optimizer=optimizer_discriminator, loss=self.loss_function_GD)
    

        return encoder, decoder, autoencoder, discriminator, generator
    
    def train_step(self, train_set, val_set, batch_size):
        x_idx_list = sample(range(train_set.shape[0]), batch_size)
        x_idx_list2 = sample(range(train_set.shape[0]), batch_size)
        x = train_set[x_idx_list]
        x2 = train_set[x_idx_list2]

        autoencoder_history = self.autoencoder.train_on_batch(x,x)
        fake_latent = self.encoder.predict(x)
        fake_image = self.decoder.predict(fake_latent)
        discriminator_input = np.concatenate((fake_image, x2))
        discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))

        discriminator_history = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)
        generator_history = self.generator.train_on_batch(x, np.ones((batch_size, 1)))

        val_x = val_set[sample(range(val_set.shape[0]), batch_size*2)]
        val_loss, val_acc = self.autoencoder.test_on_batch(val_x, val_x, reset_metrics=True, return_dict=False)

        history = {
                    'AE_loss':autoencoder_history[0],
                    'AE_acc':autoencoder_history[1],
                    'D_loss':discriminator_history, 
                    'G2_loss':generator_history,
                    'val_loss':val_loss,
                    "val_acc":val_acc
                }
        
        return history

    def train(self, train_set, batch_size, n_epochs, n_sample, logdir=r"data/Gan_training/log"):

        autoencoder_losses = []
        discriminator_losses = []
        generator_losses = []

        for epoch in np.arange(1, n_epochs):
            x_idx_list = sample(range(n_sample), batch_size)
            x = train_set[x_idx_list]

            autoencoder_history = self.autoencoder.train_on_batch(x,x)
            fake_latent = self.encoder.predict(x)
            fake_image = self.decoder.predict(fake_latent)
            discriminator_input = np.concatenate((fake_image, x))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            
            discriminator_history = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)
            generator_history = self.generator.train_on_batch(x, np.ones((batch_size, 1)))
            
            autoencoder_losses.append(autoencoder_history)
            discriminator_losses.append(discriminator_history)
            generator_losses.append(generator_history)

            if epoch == 1:
                loss_min = autoencoder_history[0]
                loss_min_epoch = 1
            
            if epoch > 500 and autoencoder_history[0] < loss_min:
                loss_min = autoencoder_history[0]
                loss_min_epoch = epoch
                self.autoencoder.save(os.path.join(logdir, "autoencoder_epoch_{}.h5".format(epoch)))
                #self.discriminator.save("../GAN_log/discriminator_epoch_{}.h5".format(epoch))
                
            
            print("Epoch--{}".format(epoch))
            print("AE_loss: {:.4f}  AE_loss_min: {:.4f}  D_loss:{:.3f}   G_loss:{:.3f}   AE_mse: {:.4f}".format(
                autoencoder_history[0], loss_min, discriminator_history, generator_history, autoencoder_history[1]
                )
            )
        self.history = {'AE_loss':autoencoder_losses, 'D_loss':discriminator_losses, 'G_loss':generator_losses}
        
        return self.history

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