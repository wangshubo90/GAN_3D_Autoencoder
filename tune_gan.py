#!/urs/bin/python3.7/python
import os, glob
import logging

from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import History
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel(logging.ERROR)

# import horovod.tensorflow as hvd

#======================= Set up Horovod ======================
# comment out this chunk of code if you train with 1 gpu
# hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import SimpleITK as sitk
from model.AAE import AAE
import numpy as np
from random import sample
from datapipeline.aae_reader import data_reader_np
from functools import reduce

class AAETrainable(tune.Trainable):
    def setup(self, config, batch_size=None, epochs=None, data=None):
        
        import tensorflow as tf
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        #===============set up model================
        self.model = AAE(**config)
        self.checkpoint1 = tf.train.Checkpoint(model=self.model.autoencoder, optimizers=self.model.autoencoder.optimizer)
        self.checkpoint2 = tf.train.Checkpoint(model=self.model.generator2, optimizers=self.model.generator2.optimizer)
        self.checkpoint3 = tf.train.Checkpoint(model=self.model.discriminator2, optimizers=self.model.discriminator2.optimizer)
        self.train_set=data
        self.batch_size=batch_size
        self.n_epochs=epochs
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        def train_step():
            x_idx_list = sample(range(self.train_set.shape[0]), self.batch_size)
            x_idx_list2 = sample(range(self.train_set.shape[0]), self.batch_size)
            
            x = self.train_set[x_idx_list]
            x2 = self.train_set[x_idx_list2]
            # x, x2 = self.train_set.take(2)

            autoencoder_history = self.model.autoencoder.train_on_batch(x,x)
            fake_latent = self.model.encoder.predict(x)
            fake_image = self.model.decoder.predict(fake_latent)
            discriminator2_input = np.concatenate((fake_image, x2))
            discriminator2_labels = np.concatenate((np.zeros((self.batch_size, 1)), np.ones((self.batch_size, 1))))

            discriminator2_history = self.model.discriminator2.train_on_batch(discriminator2_input, discriminator2_labels)
            generator2_history = self.model.generator2.train_on_batch(x, np.ones((self.batch_size, 1)))

            return {
                        'AE_loss':autoencoder_history, 
                        'D_loss':discriminator2_history, 
                        'G2_loss':generator2_history
                    }

        self.train_step = train_step

    def step(self):
        history = self.train_step()
        return history

    def save_checkpoint(self, checkpoint_dir):
        self.checkpoint1.save(checkpoint_dir)
        # self.checkpoint2save(checkpoint_dir+"_G")
        # self.checkpoint3.save(checkpoint_dir+"_D")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        
        pass 

    # sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration", max_t=400, grace_period=20)

#===============set up dataset================
datapath = r'/uctgan/data/udelCT'
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

# train_set = data_reader_np(train_set, 48, 16)

def stopper(trial_id, result):
    conditions = [
        result["AE_loss"]  < 0.005 or result["training_iteration"] > 2000,
        result["AE_loss"]  > 1 and result["training_iteration"] > 300 ,
        result["AE_loss"]  > 0.1 and result["training_iteration"] > 600 
    ]
    return reduce(lambda x,y: x or y, conditions)

analysis = tune.run(
    tune.with_parameters(AAETrainable, batch_size=16, epochs=5000, data=train_set),
    name="AAE_uct",
    metric="AE_loss",
    mode="min",
    local_dir="/uctgan/data/ray_results",
    verbose=1,
    num_samples=10,
    resources_per_trial={
        "cpu": 3,
        "gpu": 1
    },
    stop=stopper, 
    config = {
                "optG_lr" : tune.loguniform(5e-6, 1e-3), 
                "optG_beta" : 0.5, 
                "optD_lr" : tune.loguniform(5e-6, 1e-3), 
                "optD_beta" : 0.5,
                "optAE_lr" : tune.loguniform(5e-6, 1e-3), 
                "optAE_beta" : 0.9
            },
    checkpoint_freq=1,
    fail_fast=True
)
print("Best hyperparameters found were: ", analysis.best_config)