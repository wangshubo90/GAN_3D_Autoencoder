#!/urs/bin/python3.7/python
import os, glob
import logging
from ray.tune.suggest.variant_generator import grid_search
import tensorflow as tf
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import History
import tensorflow as tf
tf.autograph.set_verbosity(1)
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
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.callback import Callback

from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import SimpleITK as sitk
from model.resAAE import resAAE
import numpy as np
import random
from random import sample
from datapipeline.aae_reader import data_reader_np
from functools import reduce
from sklearn.model_selection import train_test_split
from utils.losses import *

class AAETrainable(tune.Trainable):
    def setup(self, config, batch_size=None, epochs=None, data=None):
        
        import tensorflow as tf
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        #===============set up model================
        self.model = resAAE(**config)
        self.checkpoint = tf.train.Checkpoint(AE=self.model.autoencoder, AEoptimizers=self.model.autoencoder.optimizer,
            G=self.model.generator, Goptimizers=self.model.generator.optimizer,
            D=self.model.discriminator, Doptimizers=self.model.discriminator.optimizer)
        self.train_set=data["train"]
        self.val_set=data["val"]
        self.batch_size=batch_size
        self.n_epochs=epochs
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        def train_step():
            history = self.model.train_step(self.train_set, self.val_set, self.batch_size)
            return history

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
random.seed(seed)
random.shuffle(img_ls)
img_ls = img_ls[:int(0.75*len(img_ls))]
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

asha_scheduler = AsyncHyperBandScheduler(
    time_attr='training_iteration',
    max_t=3000,
    grace_period=500,
    reduction_factor=2,
    brackets=1)

analysis = tune.run(
    tune.with_parameters(AAETrainable, batch_size=16, epochs=5000, data={"train":train_set, "val":val_set}),
    name="resAAE_uct_NoDropout",
    metric="val_loss",
    mode="min",
    local_dir="/uctgan/data/ray_results",
    verbose=1,
    num_samples=4,
    scheduler=asha_scheduler,
    resources_per_trial={
        "cpu": 3,
        "gpu": 1
    },
    stop=stopper, 
    config = {
                "hidden": (32,64,128,256),
                "optAE_lr" : tune.grid_search([0.0005, 0.0001, 0.00005]), 
                "optAE_beta" : 0.9,
                "optG_lr" : tune.sample_from(lambda _:_.config.optAE_lr / np.random.randint(1,5)), 
                "optG_beta" : 0.5, 
                "optD_lr" : tune.sample_from(lambda _:_.config.optAE_lr / np.random.randint(1,5)), 
                "optD_beta" : 0.5,
                "loss_AE": "mse",
                "loss_DG": "mse",
                "acc": "mse"
            },
    keep_checkpoints_num = 20,
    checkpoint_score_attr="min-val_loss",
    checkpoint_freq=500,
    checkpoint_at_end=True,
    fail_fast=True
)
print("Best hyperparameters found were: ", analysis.best_config)