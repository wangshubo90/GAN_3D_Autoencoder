import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv1D, Conv3DTranspose, Dropout, MaxPool3D, Flatten, Reshape, Input, Permute, GlobalAvgPool3D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from random import sample

class AAE():
    #Adversarial Autoencoder
    def __init__(self, img_shape=(48, 96, 96, 1), encoded_dim=32, hidden = (32,64,128,256),batch_size=16, epochs=5000, **kwargs):
        self.encoded_dim = encoded_dim
        self.hidden=hidden
        self.batch_size=batch_size
        self.epochs=epochs
        self.optimizer_generator = Adam(kwargs["optimizer_generator_lr"], beta_1=kwargs["optimizer_generator_beta"])
        self.optimizer_discriminator = Adam(kwargs["optimizer_discriminator_lr"], beta_1=kwargs["optimizer_discriminator_beta"])
        self.optimizer_autoencoder = Adam(kwargs["optimizer_autoencoder_lr"], beta_1=kwargs["optimizer_autoencoder_beta"])
        self.img_shape = img_shape
        self.initializer = RandomNormal(mean=0., stddev=1.)
        self.encoder, self.decoder, self.autoencoder, self.discriminator, \
                self.discriminator2, self.generator, self.generator2 = self._modelCompile(
                self.img_shape, self.encoded_dim, \
                self.optimizer_autoencoder,\
                self.optimizer_discriminator,\
                self.optimizer_generator
                )

    def _buildEncoder(self, img_shape, encoded_dim):
        hidden=self.hidden
        encoder = Sequential()
        encoder.add(Conv3D(input_shape = img_shape, filters = 16, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        #encoder.add(Dropout(0.2))
        encoder.add(Conv3D(filters = hidden[0], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
        #encoder.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        
        encoder.add(Conv3D(filters = hidden[1], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
        #encoder.add(Dropout(0.2))
        encoder.add(Conv3D(filters = hidden[1], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
        #encoder.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        
        encoder.add(Conv3D(filters = hidden[2], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
        #encoder.add(Dropout(0.2))
        encoder.add(Conv3D(filters = hidden[2], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
        #encoder.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))

        encoder.add(Conv3D(filters = hidden[3], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(layers.BatchNormalization())
#         encoder.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
#         encoder.add(GlobalAvgPool3D())
#         encoder.add(Conv3D(filters = 128, kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
#         encoder.add(layers.BatchNormalization())
#         encoder.add(Conv3D(filters = 8, kernel_size=1, strides=(1,)*3, padding="SAME", activation='relu'))
#         encoder.add(layers.BatchNormalization())
#         encoder.add(Flatten())
#         encoder.add(Dense(512, activation="relu"))
#         encoder.add(Dropout(0.3))
#         encoder.add(Dense(512, activation="relu"))
#         encoder.add(Dropout(0.3))
#         encoder.add(Dense(encoded_dim))
        
        return encoder

    def _buildDecoder(self, encoded_dim):
        hidden=self.hidden
        decoder = Sequential()
#         decoder.add(Dense(512, activation='relu', input_dim=encoded_dim))
#         #decoder.add(Dropout(0.3))
#         decoder.add(Dense(512, activation='relu'))
#         #decoder.add(Dropout(0.3))
#         decoder.add(Dense(6*12*12, activation='relu'))
#         #decoder.add(Dropout(0.3))
#         #decoder.add(Reshape([12*24*24,1]))
#         #decoder.add(Conv1D(filters = 64, kernel_size=1, padding='SAME', activation='relu'))
#         #decoder.add(Permute((2,1)))
#         decoder.add(Reshape([6,12,12,1]))
        decoder.add(Conv3DTranspose(filters=hidden[-1], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-2], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-2], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-3], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-3], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-4], kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=hidden[-4], kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(layers.BatchNormalization())
        decoder.add(Conv3DTranspose(filters=1, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        
        return decoder

    def _buildDiscriminator(self, encoded_dim):

        discriminator = Sequential()
        discriminator.add(Dense(512, input_dim=encoded_dim, activation="relu"))
        discriminator.add(Dense(512, activation="relu"))
        discriminator.add(Dense(64, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))

        return discriminator
    
    def _buildDiscriminator2(self, img_shape):
        
        discriminator = Sequential()
        discriminator.add(Conv3D(input_shape = img_shape, filters = 16, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        #discriminator.add(Dropout(0.2))
        #discriminator.add(Conv3D(filters = 16, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        
        #discriminator.add(Conv3D(filters = 32, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        #discriminator.add(Dropout(0.2))
        discriminator.add(Conv3D(filters = 32, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        
        #discriminator.add(Conv3D(filters = 64, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        #discriminator.add(Dropout(0.2))
        discriminator.add(Conv3D(filters = 64, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))

        discriminator.add(Conv3D(filters = 128, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        #discriminator.add(MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        discriminator.add(GlobalAvgPool3D())
        #discriminator.add(Conv3D(filters = 1, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(Flatten())
        discriminator.add(Dense(128, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))
        
        return discriminator

    def _modelCompile(self, img_shape, encoded_dim, optimizer_autoencoder, optimizer_discriminator, optimizer_generator):

        encoder=self._buildEncoder(img_shape, encoded_dim)
        decoder=self._buildDecoder(encoded_dim)
        
        autoencoder_input = Input(shape = img_shape) # this is the input for autoencoder and main generator
        decoder_input=Input(shape=encoded_dim)
        
        autoencoder=Model(autoencoder_input, decoder(encoder(autoencoder_input)))
        autoencoder.compile(optimizer=optimizer_autoencoder, loss="mse")
        
        discriminator = None
        generator = None
#         discriminator=self._buildDiscriminator(encoded_dim)
#         discriminator.trainable = False
#         generator = Model(autoencoder_input, discriminator(encoder(autoencoder_input)))
#         generator.compile(optimizer=optimizer_generator, loss="mse")
#         discriminator.trainable = True
#         discriminator.compile(optimizer=optimizer_discriminator, loss="mse")
    
        discriminator2 = self._buildDiscriminator2(img_shape)
        discriminator2.trainable = False
        generator2=Model(autoencoder_input, discriminator2(autoencoder(autoencoder_input)))
        generator2.compile(optimizer=optimizer_generator, loss="mse")
        discriminator2.trainable = True
        discriminator2.compile(optimizer=optimizer_discriminator, loss="mse")
        
        return encoder, decoder, autoencoder, discriminator, discriminator2, generator, generator2

    def train(self, train_set, batch_size, n_epochs, n_sample):

        autoencoder_losses = []
        discriminator_losses = []
        discriminator2_losses = []
        generator_losses = []
        generator2_losses = []

        for epoch in np.arange(1, n_epochs):
            x_idx_list = sample(range(n_sample), batch_size)
            x = train_set[x_idx_list]

            autoencoder_history = self.autoencoder.train_on_batch(x,x)
            fake_latent = self.encoder.predict(x)
            fake_image = self.decoder.predict(fake_latent)
            
#             discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, self.encoded_dim)))
#             discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            
            discriminator2_input = np.concatenate((fake_image, x))
            discriminator2_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            
            discriminator_history=0
            generator_history=0
            
            #discriminator_history = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)
            #generator_history = self.generator.train_on_batch(x, np.ones((batch_size, 1)))
            
            discriminator2_history = self.discriminator2.train_on_batch(discriminator2_input, discriminator2_labels)
            generator2_history = self.generator2.train_on_batch(x, np.ones((batch_size, 1)))
            
            autoencoder_losses.append(autoencoder_history)
            discriminator_losses.append(discriminator_history)
            discriminator2_losses.append(discriminator2_history)
            generator_losses.append(generator_history)
            generator2_losses.append(generator2_history)
            
            if epoch == 1:
                loss_min = autoencoder_history
                loss_min_epoch = 1
            
            if epoch > 5 and autoencoder_history < loss_min:
                loss_min = autoencoder_history
                loss_min_epoch = epoch
                self.autoencoder.save("../GAN_log/autoencoder_epoch_{}.h5".format(epoch))
                #self.discriminator.save("../GAN_log/discriminator_epoch_{}.h5".format(epoch))
                
            
            print("Epoch--{}".format(epoch))
            print("AE_loss: {:.4f}  AE_loss_min: {:.4f}  D1_loss:{:.3f}   D2_loss:{:.3f}   G1_loss:{:.3f}   G2_loss:{:.3f}".format(
                autoencoder_history, loss_min, discriminator_history, discriminator2_history, generator_history, generator2_history
                )
            )

        self.history = {
                        'AE_loss':autoencoder_losses, 
                        'D1_loss':discriminator_losses, 
                        'D2_loss':discriminator2_losses, 
                        'G1_loss':generator_losses,
                        'G2_loss':generator2_losses
                       }
        print("Min_loss at epoch: {}".format(loss_min_epoch))
        print("Best model saved at: ../GAN_log/autoencoder_epoch_{}.h5".format(loss_min_epoch))
        
        return self.history

    def train_step(self, train_set):
        x_idx_list = sample(range(train_set.shape[0]), batch_size)
        x_idx_list2 = sample(range(train_set.shape[0]), batch_size)
        x = train_set[x_idx_list]
        x2 = train_set[x_idx_list2]

        autoencoder_history = self.autoencoder.train_on_batch(x,x)
        fake_latent = self.encoder.predict(x)
        fake_image = self.decoder.predict(fake_latent)
        discriminator2_input = np.concatenate((fake_image, x2))
        discriminator2_labels = np.concatenate((np.zeros((self.batch_size, 1)), np.ones((self.batch_size, 1))))

        discriminator2_history = self.discriminator2.train_on_batch(discriminator2_input, discriminator2_labels)
        generator2_history = self.generator2.train_on_batch(x, np.ones((self.batch_size, 1)))

        return {
                    'AE_loss':autoencoder_history, 
                    'D_loss':discriminator2_history, 
                    'G2_loss':generator2_history
                }

if __name__=="__main__":

    import SimpleITK as sitk 
    import glob, os

    seed=42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    #===============set up dataset================
    datapath = r'./data/udelCT'
    file_reference = r'./data/udelCT/File_reference.csv'

    img_ls = glob.glob(os.path.join(datapath, "*.nii*"))
    train_set = np.zeros(shape=[len(img_ls), 48, 96, 96, 1])

    idx = 0
    for file in img_ls:
        img = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img)
        img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
        train_set[idx] = img
        idx += 1

    config = {
                        "optG_lr" : 0.0005, 
                        "optG_beta" : 0.5, 
                        "optD_lr" : 0.0001, 
                        "optD_beta" : 0.5,
                        "optAE_lr" : 0.0001, 
                        "optAE_beta" : 0.9
                    }

    model = AAE(**config)

    batch_size=16
    n_epochs=6000
    seed=42
    np.random.seed(42)

    history = model.train(train_set, batch_size, n_epochs, len(img_ls))
    #print(model.train_step(train_set))