#!/home/spl/ml/sitk/bin/python
import os, json
import numpy as np
from random import sample, seed
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import backend as bk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, ConvLSTM3D, Conv3D, Masking, SpatialDropout3D, Bidirectional
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers.core import SpatialDropout2D
from utils.layers import *
from utils.mask import compute_mask
from model.resAAETF import resAAE
import matplotlib.pyplot as plt

class temporalAAEv5():
    #Adversarial Autoencoder
    def __init__(self, 
                AAE_config="",
                AAE_checkpoint=None,
                D_checkpoint=None,
                lstm_layers=[32,32,32],
                last_conv=True,
                loss_function_temporal="mse",
                **kwargs):
        self.last_conv=last_conv
        self.lstm_layers=lstm_layers
        self.AAE = resAAE(**AAE_config)
        if AAE_checkpoint:
            self.AAE.autoencoder.load_weights(AAE_checkpoint)
        if D_checkpoint:
            self.AAE.discriminator.load_weights(D_checkpoint)
        self.encoder = self.AAE.encoder
        self.decoder = self.AAE.decoder
        self.discriminator = self.AAE.discriminator
        self.temporalModel, self.aeModel = self._buildModels(self.lstm_layers, seq_length=4)
        self.loss_function_AE = self.AAE.loss_function_AE
        self.loss_function_GD = self.AAE.loss_function_GD
        self.loss_function_temporal = loss_function_temporal
        self.acc_function = self.AAE.acc_function
        self.gfactor=self.AAE.gfactor
        self.optimizer_autoencoder = self.AAE.optimizer_autoencoder
        self.optimizer_discriminator = self.AAE.optimizer_discriminator

    def _buildModels(self,lstm_hidden_layers=[32,32], seq_length=4):

        input = Input(shape=(seq_length, *self.AAE.img_shape), dtype=tf.float32)
        mask = compute_mask(input, mask_value=0.0, reduce_axes=[2,3,4,5], keepdims=False)
        x = Lambda(lambda a: bk.reshape(a, (-1, *self.AAE.img_shape)))(input)
        x = self.encoder(x, training=True)
        x2 = self.decoder(x, training=True)
        # x2 = Lambda(lambda a: bk.reshape(a, (*input.shape)))(x2)
        x = SpatialDropout3D(0, data_format="channels_last")(x)
        ax = Conv3D(filters=x.shape[-1], kernel_size=1, strides=(1,1,1), padding="same")(x)
        ax = tf.keras.activations.softmax(ax)
        x = x*ax
        x = BatchNormalization()(x)
        
        x = Lambda(lambda a: bk.reshape(a, (-1, seq_length, *self.encoder.output_shape[1:])))(x)
        # model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (1, -1, *self.encoder.output_shape[1:]) )))
        for i in lstm_hidden_layers:
            x =  Bidirectional(ConvLSTM3D(i,3,padding="same",activation="tanh", return_sequences=True))(x, mask=mask)
            x = BatchNormalization()(x)
        x = ConvLSTM3D(self.encoder.output_shape[-1], 3, padding="same", activation="tanh")(x, mask=mask)
        # x = spatialLSTM3D(x, mask = mask, lstm_activation="tanh")
        x = Lambda(lambda a: bk.reshape(a, (-1, *self.decoder.input_shape[1:])))(x)
        x = BatchNormalization()(x)
        x = self.decoder(x, training=False)
        temporalModel = Model(inputs=input, outputs=x)
        aeModel = Model(inputs=input, outputs=x2)
        # model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (-1, *self.encoder.output_shape[1:]) )))
        return temporalModel, aeModel

    # @tf.function
    def __train_step__(self, x, y, aug_x, val_x, val_y, val_aug_x,  gfactor, validate=True):
        input = x
        input_aug = aug_x
        seqlength = x.shape[1]
        with tf.GradientTape() as ae_tape:
            fake_image_temporal = self.temporalModel(aug_x, training=True)
            fake_image_ae = self.aeModel(aug_x, training=True)
            fake_image = tf.concat([fake_image_temporal, fake_image_ae], axis=0)
            # aug_x = Lambda(lambda a: bk.reshape(a, (-1, *self.AAE.img_shape)))(aug_x)
            x = Lambda(lambda a: bk.reshape(a, (-1, *self.AAE.img_shape)))(x)
            y_x = tf.concat([y, x], axis=0)
            fake_output = self.discriminator(fake_image, training=False)
            ae_loss = self.loss_function_AE(x, fake_image_ae)
            temporal_loss = self.loss_function_temporal(y, fake_image_temporal)
            g_loss = self.loss_function_GD(fake_output, tf.ones_like(fake_output))
            total_loss = 0*ae_loss + temporal_loss + g_loss * gfactor
            ae_acc = self.acc_function(x, fake_image_ae)
            temporal_acc = self.acc_function(y, fake_image_temporal)
        train_output=[fake_image, y_x]
        ae_grad = ae_tape.gradient(total_loss, self.temporalModel.trainable_variables)
        self.optimizer_autoencoder.apply_gradients(zip(ae_grad, self.temporalModel.trainable_variables))
        x = input
        aug_x = input_aug
        with tf.GradientTape() as disc_tape:
            # fake_image_temporal = self.temporalModel(aug_x, training=False)
            # fake_image_ae = self.aeModel(aug_x, training=False)
            # fake_image = tf.concat([fake_image_temporal, fake_image_ae], axis=0)
            fake_output = self.discriminator(fake_image, training=True)
            real_output = self.discriminator(y_x, training=True)

            d_label = tf.concat([tf.zeros_like(fake_output), tf.ones_like(real_output)], axis=0)
            d_loss = self.loss_function_GD(tf.concat([fake_output, real_output], axis=0), d_label)

        d_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        if validate:
            val_output_temporal = self.temporalModel(val_x, training = False)
            val_output_ae = self.aeModel(val_x, training=False)
            val_output = tf.concat([val_output_temporal, val_output_ae], axis=0)
            val_x = Lambda(lambda a: bk.reshape(a, (-1, *self.AAE.img_shape)))(val_x)
            val_yx = tf.concat([val_y, val_x], axis=0)
            val_loss = self.loss_function_AE(val_x, val_output_ae)+self.loss_function_AE(val_y, val_output_temporal)
            val_ae_acc = self.acc_function(val_x, val_output_ae)
            val_temporal_acc = self.acc_function(val_y, val_output_temporal)
            return [ae_loss, d_loss, g_loss, total_loss, ae_acc, temporal_acc, val_loss, val_ae_acc, val_temporal_acc], [val_output, val_yx], train_output
        else:
            return ae_loss, ae_acc, d_loss, g_loss, total_loss
        
    def __fetch_batch__(self,dataset):
        
        return next(dataset)
    
    def train_step(self, train_set, val_set, validate=True):
        # x_idx_list = sample(range(train_set.shape[0]), batch_size)
        # x_idx_list2 = sample(range(train_set.shape[0]), batch_size)
        train_batch = self.__fetch_batch__(train_set)
        val_batch = self.__fetch_batch__(val_set)

        if validate:
            [ae_loss, d_loss, g_loss, total_loss, ae_acc, temporal_acc, val_loss, val_ae_acc, val_temporal_acc], val_output, train_output= self.__train_step__(*train_batch, *val_batch, self.gfactor, validate=validate)
            
            history = {
                        'AE_loss':ae_loss,
                        'total_loss': total_loss,
                        'D_loss':d_loss, 
                        'G_loss':g_loss,
                        'AE_acc':ae_acc,
                        'Temporal_acc':temporal_acc,
                        'val_loss':val_loss,
                        "val_ae_acc":val_ae_acc,
                        "val_temp_acc":val_temporal_acc
                    }
            
            return history, val_output, train_output
        else:
            ae_loss, ae_acc, d_loss, g_loss, total_loss = self.__train_step__(*train_batch, *val_batch, self.gfactor, validate=validate)
            
            history = {
                        'AE_loss':ae_loss,
                        'total_loss': total_loss,
                        'AE_acc':ae_acc,
                        'D_loss':d_loss, 
                        'G_loss':g_loss,
                    }
            
            return history

    def train(self, train_set, val_set, n_epochs, logdir=r"data/Gan_training/log", logstart=500, logfreq=500, logimage=8, slices=None):
        summary_writer = tf.summary.create_file_writer(logdir)
        summary_writer.set_as_default()
        for epoch in np.arange(1, n_epochs):
            history, val_output, train_output = self.train_step(train_set, val_set)
            with summary_writer.as_default():
                for k, v in history.items():
                    tf.summary.scalar(k, data=v, step=epoch)
            if epoch == 1:
                loss_min = history["val_temp_acc"]
                loss_min_epoch = 1
                summary = {k:[v] for k,v in history.items()}
            # else:
            #     summary = {k:v.append(history[k]) for k,v in summary.items()}
            
            if epoch > logstart and bool(epoch % logfreq == 0 or history["val_temp_acc"] < loss_min):
                if history["val_temp_acc"] < loss_min:
                    loss_min = history["val_temp_acc"]
                    loss_min_epoch = epoch
                checkpointDir = os.path.join(logdir, f"epoch-{epoch}")
                os.makedirs(checkpointDir, exist_ok=True)
                self.temporalModel.save_weights(os.path.join(checkpointDir, "tAAE_epoch_{}.h5".format(epoch)))
                if logimage:
                    self.save_image(val_output, epoch, logdir=checkpointDir, img_name="val-image", logimage=logimage, slices = slices)
                    self.save_image(train_output, epoch, logdir=checkpointDir, img_name="train-image", logimage=logimage, slices = slices)
                #self.discriminator.save("../GAN_log/discriminator_epoch_{}.h5".format(epoch))
            
            print("Epoch -- {} -- CurrentBest -- {} -- val-acc -- {:.4f}".format(epoch, loss_min_epoch, loss_min))
            
            print("   ".join(["{}: {:.4f}".format(k, v) for k, v in history.items()])
            )
            
        return summary

    def __plot_image__():
        pass

    def __tsbd_log__(self, writer, metrics, image=None):
        pass
    
    def save_image(self, output, epoch, logdir=".", img_name="val-image", logimage=8, slices=None):
        image = np.squeeze(output[0].numpy())[:logimage]*255
        valimage = np.squeeze(output[1])[:logimage]*255

        image = image.astype(np.uint8)
        valimage = valimage.astype(np.uint8)

        shape = image.shape
        if slices:
            image = image[slices]
            valimage = valimage[slices]
        else:
            image = image[:, shape[1]//2, ...]
            valimage = valimage[:, shape[1]//2, ...]
        image = np.concatenate([image.reshape((-1, shape[-1])), valimage.reshape((-1, shape[-1]))], axis=1)

        fig = plt.figure()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "{}-{}.png".format(img_name, epoch)), dpi=300)
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