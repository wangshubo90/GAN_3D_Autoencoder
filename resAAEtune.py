#!/urs/bin/python3.7/python
import os, glob
import logging
from ray.tune.suggest.variant_generator import grid_search

from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import History
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel(logging.ERROR)

<<<<<<< HEAD
from tqdm import tqdm


physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

config={
    "optG_lr":0.0002,
    "optG_beta":0.5,
    "optD_lr":0.0001,
    "optD_beta":0.5,
    "optAE_lr":0.001,
    "optAE_beta":0.9,
    "img_shape": (48, 96, 96, 1), 
    "encoded_dim": 16, 
    "loss": "mse", 
    "acc": "mse",
    "hidden": (16, 32, 64, 128),
    "output_slices": slice(None),
    "batch_size": 16,
    "epochs": 5000
}
#[slice(None), slice(None,15), slice(2,62), slice(2,62), slice(None)]
model = resAAE(**config)
    
datapath = r'../Data'
file_reference = r'../Training/File_reference.csv'

img_ls = os.listdir(datapath)
train_set = np.zeros(shape=[len(img_ls), 48, 96, 96, 1])

idx = 0
for file in tqdm(img_ls):
    img = sitk.ReadImage(os.path.join(datapath, file))
    img = sitk.GetArrayFromImage(img)
    img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
    train_set[idx] = img
    idx += 1

batch_size=12
n_epochs=1000
seed=42
np.random.seed(42)

history = model.train(train_set, batch_size, n_epochs, len(img_ls))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(16,12))

ax[0].plot(range(2999), history["AE_loss"], label="AE_loss")
ax[0].title("AE_loss")

fig, ax = plt.subplots(2,2)
=======
# import horovod.tensorflow as hvd
>>>>>>> 23e44c830f6b30f3c92e4f16fc9a7f360944f8b2

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
from ray.tune.callback import Callback

from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import SimpleITK as sitk
from model.AAE import AAE
import numpy as np
from random import sample
from datapipeline.aae_reader import data_reader_np
from functools import reduce
from sklearn.model_selection import train_test_split

class AAETrainable(tune.Trainable):
    def setup(self, config, batch_size=None, epochs=None, data=None):
        
        import tensorflow as tf
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        #===============set up model================
        self.model = AAE(**config)
        self.checkpoint = tf.train.Checkpoint(AE=self.model.autoencoder, AEoptimizers=self.model.autoencoder.optimizer,
            GeneratorExit=self.model.generator2, Goptimizers=self.model.generator2.optimizer,
            D=self.model.discriminator2, Doptimizers=self.model.discriminator2.optimizer)
        self.train_set=data["train"]
        self.val_set=data["val"]
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
            
            val_x = self.val_set[sample(range(self.val_set.shape[0]), self.batch_size*2)]
            val_loss = self.model.autoencoder.test_on_batch(val_x, val_x, reset_metrics=True, return_dict=False)

            return {
                        'AE_loss':autoencoder_history, 
                        'D_loss':discriminator2_history, 
                        'G2_loss':generator2_history,
                        'val_loss':val_loss
                    }
        self.val_loss = 0.02
        self.currentIsBest=False
        self.train_step = train_step

    def step(self):
        history = self.train_step()
        if history["val_loss"] < self.val_loss:
            self.val_loss = history["val_loss"]
            self.currentIsBest = True
        else:
            self.currentIsBest = False
        return history

    def save_checkpoint(self, checkpoint_dir):
        if self.currentIsBest:
            self.model.autoencoder.save(os.path.join(checkpoint_dir,"AE.h5"))
        self.checkpoint.save(checkpoint_dir)
        # self.checkpoint2save(checkpoint_dir+"_G")
        # self.checkpoint3.save(checkpoint_dir+"_D")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        
        pass 

    # sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration", max_t=400, grace_period=20)

#===============set up dataset================
def read_data(file_ls):
    dataset = np.zeros(shape=[len(file_ls), 48, 96, 96, 1], dtype=np.float32)
    
    for idx, file in enumerate(file_ls):
        img = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img)
        img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
        dataset[idx] = img
    return dataset

datapath = r'/uctgan/data/udelCT'
file_reference = r'./data/udelCT/File_reference.csv'
seed=42
img_ls = glob.glob(os.path.join(datapath, "*.nii*"))
train_img, test_img = train_test_split(img_ls, test_size=0.3, random_state=seed)
val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

train_set = read_data(train_img) 
val_set = read_data(val_img)

def stopper(trial_id, result):
    conditions = [
        result["AE_loss"]  < 0.001 or result["training_iteration"] > 3000,
        result["AE_loss"]  > 1 and result["training_iteration"] > 250 ,
        result["AE_loss"]  > 0.2 and result["training_iteration"] > 500 ,
    ]
    return reduce(lambda x,y: x or y, conditions)

class MyCallback(Callback):
    def on_trial_complete(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['metric']}")

analysis = tune.run(
    tune.with_parameters(AAETrainable, batch_size=12, epochs=5000, data={"train":train_set, "val":val_set}),
    name="AAE_uct_test3",
    metric="val_loss",
    mode="min",
    local_dir="/uctgan/data/ray_results",
    verbose=1,
    num_samples=6,
    resources_per_trial={
        "cpu": 3,
        "gpu": 1
    },
    stop=stopper, 
    config = {
                "hidden":(32,64,128,256),
                "optG_lr" : tune.uniform(5e-6, 1e-4), 
                "optG_beta" : 0.5, 
                "optD_lr" : tune.sample_from(lambda spec: spec.config.optG_lr/np.random.randint(5, 20)), 
                "optD_beta" : 0.5,
                "optAE_lr" : 0.0005, 
                "optAE_beta" : 0.9,
                "act": "relu"
            },
    keep_checkpoints_num = 100,
    checkpoint_score_attr="min-val_loss",
    checkpoint_freq=1,
    checkpoint_at_end=True,
    fail_fast=True
)
print("Best hyperparameters found were: ", analysis.best_config)