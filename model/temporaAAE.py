#!/home/spl/ml/sitk/bin/python
import os
import numpy as np
from random import sample, seed
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import backend as bk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, ConvLSTM3D, Conv3D
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from utils.layers import *
from model.resAAETF import resAAE
import matplotlib.pyplot as plt

class temporalAAE():
    #Adversarial Autoencoder
    def __init__(self, 
                AAE_config="",
                AAE_checkpoint="",
                lstm_layers=[32,32,32],
                last_conv=True,
                **kwargs):
        self.last_conv=last_conv
        self.lstm_layers=lstm_layers
        self.AAE = resAAE(**AAE_config)
        self.AAE.autoencoder.load_weights(AAE_checkpoint)
        self.encoder = self.AAE.encoder
        self.decoder = self.AAE.decoder
        self.temporalModel = self._buildTemporal(self.lstm_layers, self.last_conv)

    def _buildTemporal(self,lstm_hidden_layers=[32,32], last_conv=True):

        model = Sequential()
        model.add(Input(shape=(None), dtype=tf.float32, ragged=True))
        model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (1, -1, *self.encoder.output_shape[1:]) )))
        for i in lstm_hidden_layers:
            model.add(ConvLSTM3D(i,3,padding="same",activation="tanh", return_sequences=True))
        model.add(ConvLSTM3D(self.encoder.output_shape[-1], 3, padding="same", activation="relu"))
        model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (-1, *self.encoder.output_shape[1:]) )))
        return model

    @tf.function
    def __train_step__(self, x, val_x, gfactor, validate=True):
        
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

    def __plot_image__():
        pass

    def __tsbd_log__(self, writer, metrics, image=None):
        pass
    
    def save_image(self, output, epoch, logdir=".", logimage=8):
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