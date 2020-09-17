import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Conv3DTranspose
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from tensorflow.keras.initializers import RandomNormal

class GAE():
    def __init__(self, img_shape=(48, 96, 96, 1), encoded_dim=8, optimizer = SGD(0.001, momentum=.9), 
                optimizer_discriminator = SGD(0.0001, momentum=.9), optimizer_autoencoder = Adam(0.0001)):
        self.encoded_dim = encoded_dim
        self.optimizer = optimizer
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_autoencoder = optimizer_autoencoder
        self.img_shape = img_shape
        self.initializer = RandomNormal(mean=0., stddev=1.)
        self._initAndCompileFullModel(img_shape, encoded_dim)


    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = Sequential()
        encoder.add(keras.layers.Conv3D(input_shape = img_shape, filters = 16, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        encoder.add(keras.layers.Conv3D(input_shape = img_shape, filters = 32, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        encoder.add(keras.layers.Conv3D(input_shape = img_shape, filters = 64, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        encoder.add(keras.layers.Conv3D(input_shape = img_shape, filters = 128, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        encoder.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        encoder.add(keras.layers.GlobalAvgPool3D())
        encoder.add(keras.layers.Flatten())
        encoder.add(Dense(encoded_dim))
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = Sequential()
        decoder.add(Dense(128, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(3*6*6*128, activation='relu'))
        decoder.add(Reshape([3,6,6,128]))
        decoder.add(Conv3DTranspose(filters=64, kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(Conv3DTranspose(filters=32, kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(Conv3DTranspose(filters=16, kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))
        decoder.add(Conv3DTranspose(filters=1, kernel_size=3, strides=(2,)*3, padding="SAME", activation='relu'))

        #decoder.add(Dense(1000, activation='relu'))
        #decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.summary()
        return decoder

    def _getDescriminator(self, img_shape):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        discriminator = Sequential()
        discriminator.add(keras.layers.Conv3D(input_shape = img_shape, filters = 16, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        discriminator.add(keras.layers.Conv3D(input_shape = img_shape, filters = 32, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        discriminator.add(keras.layers.Conv3D(input_shape = img_shape, filters = 64, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        discriminator.add(keras.layers.Conv3D(input_shape = img_shape, filters = 128, kernel_size=3, strides=(1,)*3, padding="SAME", activation='relu'))
        discriminator.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))
        discriminator.add(keras.layers.GlobalAvgPool3D())
        discriminator.add(keras.layers.Flatten())
        discriminator.add(Dense(32, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))
        discriminator.summary()
        return discriminator

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(img_shape)
        img = Input(shape=img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        self.autoencoder.compile(optimizer=self.optimizer_autoencoder, loss='mse')
        self.discriminator.compile(optimizer=self.optimizer_discriminator, 
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        for layer in self.discriminator.layers:
            layer.trainable = False

        is_real = self.discriminator(gen_img)
        self.autoencoder_discriminator = Model(img, is_real)
        self.autoencoder_discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy',
                                           metrics=['accuracy'])

    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        for i in range(-5, 5):
            for j in range(-5,5):
                topred = np.array((i*0.5,j*0.5))
                topred = topred.reshape((1, 2))
                img = self.decoder.predict(topred)
                img = img.reshape(self.img_shape)
                ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        fig.savefig(str(epochnumber)+".png")
        plt.show()
        plt.close(fig)

    def train(self, x_train, batch_size=4, epochs=5):
        self.autoencoder.fit(x_train, x_train, epochs=1)
        for epoch in range(epochs):
            #---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs_real = x_train[idx]
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs_real2 = x_train[idx]
            # Generate a half batch of new images
            #gen_imgs = self.decoder.predict(latent_fake)
            imgs_fake = self.autoencoder.predict(imgs_real2)
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            #d_loss = (0,0)
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs_real = x_train[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))
            # Train generator
            g_logg_similarity = self.autoencoder_discriminator.train_on_batch(imgs_real, valid_y)
            # Plot the progress
            print ("%d [D loss = %.2f, accuracy: %.2f] [G loss = %f, accuracy: %.2f]" % (epoch,d_loss[0], d_loss[1], g_logg_similarity[0], g_logg_similarity[1]))
#            if(epoch % save_interval == 0):
#                self.imagegrid(epoch)
        codes = self.encoder.predict(x_train)
#        params = {'bandwidth': [3.16]}#np.logspace(0, 2, 5)}
#        grid = GridSearchCV(KernelDensity(), params, n_jobs=4)
#        grid.fit(codes)
#        print grid.best_params_
#        self.kde = grid.best_estimator_
        self.kde = KernelDensity(kernel='gaussian', bandwidth=3.16).fit(codes)


    def generate(self, n = 10000):
        codes = self.kde.sample(n)
        images = self.decoder.predict(codes)
        return images

    def generateAndPlot(self, x_train, n = 10, fileName="generated.png"):
        fig = plt.figure(figsize=[20, 20])
        images = self.generate(n*n)
        index = 1
        for image in images:
            image = image.reshape(self.img_shape)
            ax = fig.add_subplot(n, n+1, index)
            index=index+1
            ax.set_axis_off()
            ax.imshow(image, cmap="gray")
            if((index)%(n+1) == 0):
                nearest = helpers.findNearest(x_train, image)
                ax = fig.add_subplot(n, n+1, index)
                index= index+1
                ax.imshow(nearest, cmap="gray")
        fig.savefig(fileName)
        plt.show()

    def meanLogLikelihood(self, x_test):
        KernelDensity(kernel='gaussian', bandwidth=0.2).fit(codes)

if __name__ == "__main__":
    sample = np.ones(shape = (4, 48, 96, 96, 1))
    model = GAE()
    model.train(sample, batch_size=4, epochs=5)