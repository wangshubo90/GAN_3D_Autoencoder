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
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, ConvLSTM3D, Conv3D, Masking, SpatialDropout3D
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers.core import Dropout, SpatialDropout2D
from tensorflow.python.ops.gen_batch_ops import Batch
from utils.layers import *
from utils.mask import compute_mask
from model.resAAETF import resAAE
import matplotlib.pyplot as plt

class temporalAAEv4():
    #Adversarial Autoencoder
    def __init__(self, 
                AAE_config="",
                AAE_checkpoint="",
                D_checkpoint=None,
                batch_size=8,
                lstm_layers=[32,32,32],
                last_conv=True,
                **kwargs):
        self.batch_size = batch_size
        self.last_conv=last_conv
        self.lstm_layers=lstm_layers
        self.AAE = resAAE(**AAE_config)
        self.AAE.autoencoder.load_weights(AAE_checkpoint)
        if D_checkpoint:
            self.AAE.discriminator.load_weights(D_checkpoint)
        lstmskiplayers = [self.AAE.encoder.get_layer(name=layername) for layername in [
                "tf.nn.relu_2", "tf.nn.relu_4", "tf.nn.relu_6"
            ]]
        self.encoder = Model(inputs=self.AAE.encoder.input, outputs=[layer.output for layer in lstmskiplayers])
        self.decoder = self.AAE.decoder
        self.discriminator = self.AAE.discriminator
        self.temporalModel = self._buildTemporal(self.lstm_layers, seq_length=4)
        self.loss_function_AE = self.AAE.loss_function_AE
        self.loss_function_GD = self.AAE.loss_function_GD
        self.acc_function = self.AAE.acc_function
        self.gfactor=self.AAE.gfactor
        self.optimizer_autoencoder = self.AAE.optimizer_autoencoder
        self.optimizer_discriminator = self.AAE.optimizer_discriminator

    def _buildTemporal(self, lstm_hidden_layers=[32,32], seq_length=4):
        
        input = Input(shape=(seq_length, *self.AAE.img_shape), dtype=tf.float32, batch_size=self.batch_size)
        input2 = Input(shape=(seq_length, seq_length), dtype=tf.float32, batch_size=self.batch_size)
        adj = bk.reshape(input2, shape=(self.batch_size,1,1,1,seq_length*seq_length))
        adj = tf.pad(adj, [[0,0], [1,1], [1,1], [1,1], [0,0]], mode="SYMMETRIC")
        adj = tf.pad(adj, [[0,0], [0,0], [1,2], [1,2], [0,0]], mode="SYMMETRIC")
        adj = bk.reshape(adj, shape=(self.batch_size,3,6,6,seq_length,seq_length))
        # mask = compute_mask(input, mask_value=0.0, reduce_axes=[2,3,4,5], keepdims=False)
        x = Lambda(lambda a: bk.reshape(a, (-1, *self.AAE.img_shape)))(input)
        x1,x2,x3 = self.encoder(x, training=False)
        # x = SpatialDropout3D(0.3, data_format="channels_last")(x)
        x1, x2, x3 = [Lambda(lambda a: bk.reshape(a, (-1, seq_length, *a.shape[1:])))(x_) for x_ in [x1,x2,x3]]
        # model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (1, -1, *self.encoder.output_shape[1:]) )))
        x3 = GCNfuncLayer(x3, adj, channel_out = x3.shape[-1]*2, seqToGraph=True, activation="relu")
        x3 = GCNfuncLayer(x3, adj, channel_out = x3.shape[-1]*4, seqToGraph=False, activation="relu")
        # x3 = Dropout(0.2)(x3)
        x3 = GCNfuncLayer(x3, adj, channel_out = self.encoder.output_shape[-1][-1], seqToGraph=False, activation="relu")
        # x3 = GCNfuncLayer(input, input2, seqToGraph=False, activation="relu")
        # x3 = Dropout(0.2)(x3)
        x3 = tf.transpose(x3, perm = [0,1,2,3,5,4])
        x3 = Dense(1, use_bias=True, activation="relu")(x3)
        # x3 = Dropout(0.2)(x3)
        x3 = tf.squeeze(x3, -1)
        
        x = self.decoder(x3, training=False)
        model = Model(inputs=[input, input2], outputs=x)
        # model.add(Lambda(lambda x: keras.backend.reshape(x, shape = (-1, *self.encoder.output_shape[1:]) )))
        return model

    @tf.function
    def __train_step__(self, x, y,adj, val_x, val_y, val_adj,  gfactor, validate=True):
        input = x
        seqlength = x.shape[1]
        with tf.GradientTape() as ae_tape:
            fake_image = self.temporalModel(inputs=(x, adj))
            fake_output = self.discriminator(fake_image, training=False)

            ae_loss = self.loss_function_AE(y, fake_image)
            g_loss = self.loss_function_GD(fake_output, tf.ones_like(fake_output))
            total_loss = ae_loss + g_loss*gfactor
            ae_acc = self.acc_function(y, fake_image)
        
        ae_grad = ae_tape.gradient(total_loss, self.temporalModel.trainable_variables)
        self.optimizer_autoencoder.apply_gradients(zip(ae_grad, self.temporalModel.trainable_variables))
        x = input
        with tf.GradientTape() as disc_tape:
            fake_image = self.temporalModel(inputs=(x,adj), training=False)
            fake_output = self.discriminator(fake_image, training=True)
            real_output = self.discriminator(y, training=True)

            d_label = tf.concat([tf.zeros_like(fake_output), tf.ones_like(real_output)], axis=0)
            d_loss = self.loss_function_GD(tf.concat([fake_output, real_output], axis=0), d_label)

        d_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        if validate:
            val_output = self.temporalModel(inputs=(val_x,val_adj), training = False)
            val_loss = self.loss_function_AE(val_output, val_y)
            val_acc = self.acc_function(val_output, val_y)
            return [ae_loss, ae_acc, d_loss, g_loss, val_loss, val_acc, total_loss], [val_output, val_y]
        else:
            return ae_loss, ae_acc, d_loss, g_loss, total_loss

    def train_step(self, train_set, val_set, validate=True):
        # x_idx_list = sample(range(train_set.shape[0]), batch_size)
        # x_idx_list2 = sample(range(train_set.shape[0]), batch_size)
        x, y, adj = next(train_set)
        val_x, val_y, val_adj = next(val_set)

        if validate:
            [ae_loss, ae_acc, d_loss, g_loss, val_loss, val_acc, total_loss], val_output= self.__train_step__(x, y, adj, val_x, val_y, val_adj, self.gfactor, validate=validate)
            
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
            ae_loss, ae_acc, d_loss, g_loss, total_loss = self.__train_step__(x, y, adj, val_x, val_y, val_adj, self.gfactor, validate=validate)
            
            history = {
                        'AE_loss':ae_loss,
                        'total_loss': total_loss,
                        'AE_acc':ae_acc,
                        'D_loss':d_loss, 
                        'G_loss':g_loss,
                    }
            
            return history

    def train(self, train_set, val_set, n_epochs, logdir=r"data/Gan_training/log", logstart=500, logimage=8, slices=None):
        summary_writer = tf.summary.create_file_writer(logdir)
        summary_writer.set_as_default()
        for epoch in np.arange(1, n_epochs):
            history, val_output = self.train_step(train_set, val_set)
            with summary_writer.as_default():
                for k, v in history.items():
                    tf.summary.scalar(k, data=v, step=epoch)
            if epoch == 1:
                loss_min = history["val_acc"]
                loss_min_epoch = 1
                summary = {k:[] for k in history.keys()}
            
            if epoch > logstart and history["val_acc"] < loss_min:
                loss_min = min(history["val_acc"], loss_min)
                loss_min_epoch = epoch
                self.temporalModel.save_weights(os.path.join(logdir, "tAAE_epoch_{}.h5".format(epoch)))
                if logimage:
                    self.save_image(val_output, epoch, logdir, logimage, slices)
                #self.discriminator.save("../GAN_log/discriminator_epoch_{}.h5".format(epoch))
            
            print("Epoch -- {} -- CurrentBest -- {} -- val-acc -- {:.4f}".format(epoch, loss_min_epoch, loss_min))
            
            print("   ".join(["{}: {:.4f}".format(k, v) for k, v in history.items()])
            )
            

        return summary

    def __plot_image__():
        pass

    def __tsbd_log__(self, writer, metrics, image=None):
        pass
    
    def save_image(self, output, epoch, logdir=".", logimage=8, slices=None):
        image = np.squeeze(output[0].numpy())[:logimage]
        valimage = np.squeeze(output[1])[:logimage]
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