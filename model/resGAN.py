#!/home/spl/ml/sitk/bin/python
import numpy as np
from random import sample, seed
from tqdm import tqdm
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv1D, Conv3DTranspose, Flatten, Reshape, Input, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from utils.layers import *

class AAE():
    #Adversarial Autoencoder
    def __init__(self, 
                img_shape=(48, 96, 96, 1), 
                encoded_dim=16, 
                loss = "mse", 
                acc = "mse",
                optimizer_generator = SGD(0.001, momentum=.9), 
                optimizer_discriminator = SGD(0.0001, momentum=.9), 
                optimizer_autoencoder = Adam(0.0001)):
        self.encoded_dim = encoded_dim
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_autoencoder = optimizer_autoencoder
        self.img_shape = img_shape
        self.initializer = RandomNormal(mean=0., stddev=1.)
        self.encoder, self.decoder, self.autoencoder, self.discriminator, self.generator = self._modelCompile(
                self.img_shape, self.encoded_dim, \
                self.optimizer_autoencoder,\
                self.optimizer_discriminator,\
                self.optimizer_generator
                )

    def _buildEncoder(self, img_shape, encoded_dim):
        
        input = Input(shape=(48, 96, 96, 1))
        x = Conv3D(filters=16, kernel_size=5, strides=(2,2,2), padding="SAME")(input)
        x = BatchNormalization()(x)
        x = relu(x)
        x = residual_block(x, filters = 32, kernel_size= 3,  
                    strides = (2,2,2), padding = "SAME", activate=relu)
        x = residual_block(x, filters = 64, kernel_size= 3,  
                    strides = (2,2,2), padding = "SAME", activate=relu)
        x = residual_block(x, filters = 128, kernel_size= 3,  
                    strides = (2,2,2), padding = "SAME", activate=relu)
        
        encoder = Model(inputs=input, outputs=x)        
        return encoder
     
    def _buildDecoder(self, encoded_dim):

        input = Input(shape = (3, 12, 12))
        return decoder

    def _buildDiscriminator(self, encoded_dim):

        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=encoded_dim, activation="relu"))
        discriminator.add(Dense(256, activation="relu"))
        discriminator.add(Dense(64, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))

        return discriminator

    def _modelCompile(self, img_shape, encoded_dim, optimizer_autoencoder, optimizer_discriminator, optimizer_generator):

        encoder=self._buildEncoder(img_shape, encoded_dim)
        decoder=self._buildDecoder(encoded_dim)
        
        autoencoder_input = Input(shape = img_shape)
        autoencoder=Model(autoencoder_input, decoder(encoder(autoencoder_input)))
        autoencoder.compile(optimizer=optimizer_autoencoder, loss="mse")

        discriminator=self._buildDiscriminator(encoded_dim)
        discriminator.trainable = False
        generator = Model(autoencoder_input, discriminator(encoder(autoencoder_input)))
        generator.compile(optimizer=optimizer_generator, loss="binary_crossentropy")
        discriminator.trainable = True
        discriminator.compile(optimizer=optimizer_discriminator, loss="binary_crossentropy")
    

        return encoder, decoder, autoencoder, discriminator, generator

    def train(self, train_set, batch_size, n_epochs, n_sample):

        autoencoder_losses = []
        discriminator_losses = []
        generator_losses = []

        for epoch in np.arange(1, n_epochs):
            x_idx_list = sample(range(n_sample), batch_size)
            x = train_set[x_idx_list]

            autoencoder_history = self.autoencoder.train_on_batch(x,x)
            fake_latent = self.encoder.predict(x)
            discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, self.encoded_dim)))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            
            discriminator_history = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)
            generator_history = self.generator.train_on_batch(x, np.ones((batch_size, 1)))
            
            autoencoder_losses.append(autoencoder_history)
            discriminator_losses.append(discriminator_history)
            generator_losses.append(generator_history)

            if epoch % 50 == 0:
                self.autoencoder.save("./train_log/autoencoder_epoch_{}.h5".format(epoch))
                self.discriminator.save("./train_log/autoencoder_epoch_{}.h5".format(epoch))
            print("Epoch--{}".format(epoch))
            print("AE_loss: {:.4f}  D_loss:{:.3f}   G_loss:{:.3f}".format(autoencoder_history, discriminator_history, generator_history))

        self.history = {'AE_loss':autoencoder_losses, 'D_loss':discriminator_losses, 'G_loss':generator_losses}
        
        return self.history

    def train_on_batch():
        return

    def load_model(self):

        return

    def encodeImage(self):

        return 

    def plot_losses(self):

        return

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

    model = AAE()
    
    import os
    import SimpleITK as sitk 

    datapath = r'../Data'
    file_reference = r'../training/File_reference.csv'

    img_ls = os.listdir(datapath)
    train_set = np.zeros(shape=[len(img_ls), 48, 96, 96, 1])

    idx = 0
    for file in tqdm(img_ls):
        img = sitk.ReadImage(os.path.join(datapath, file))
        img = sitk.GetArrayFromImage(img)
        img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
        train_set[idx] = img
        idx += 1

    model = AAE(encoded_dim=128)

    batch_size=12
    n_epochs=3000
    seed=42
    np.random.seed(42)

    history = model.train(train_set, batch_size, n_epochs, len(img_ls))

    import matplotlib as plt

    fig, ax = plt.subplots(2, 2, figsize=(16,12))

    ax[0].plot(range(2999), history["AE_loss"], label="AE_loss")
    ax[0].title("AE_loss")

    fig, ax = plt.subplots(2,2)

    ax[0, 0].plot(range(2999), history["AE_loss"], label="AE_loss")
    ax[0, 0].set_title("AE_loss")

    ax[0, 1].plot(range(2999), history["G_loss"], label="G_loss")
    ax[0, 1].set_title("G_loss")

    ax[1, 0].plot(range(2999), history["D_loss"], label="D_loss")
    ax[1, 0].set_title("D_loss")

    plt.show()