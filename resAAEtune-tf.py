#!/urs/bin/python3.7/python
import os, logging
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
import tensorflow as tf
tf.autograph.set_verbosity(1)
tf.get_logger().setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
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
from ray.tune.suggest.variant_generator import grid_search

import numpy as np
import random
from random import sample
import SimpleITK as sitk
from model.resAAETF import resAAE
from datapipeline.aae_reader import data_reader_np
from functools import reduce
from sklearn.model_selection import train_test_split
from utils.losses import *
from utils.filters import *
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import relu

class AAETrainable(tune.Trainable):
    def setup(self, config, batch_size=None, epochs=None, data=None):
        
        import tensorflow as tf
        seed=42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        #===============set up model================
        self.model = resAAE(**config)
        self.train_set=data["train"]
        self.val_set=data["val"]
        self.batch_size=batch_size
        self.n_epochs=epochs
        self.val_loss = 0.01
        self.currentIsBest=False

    def step(self):
        history, val_output = self.model.train_step(self.train_set, self.val_set, self.batch_size)
        self.val_output=val_output
        if history["val_loss"] < self.val_loss:
            self.val_loss = history["val_loss"]
            self.currentIsBest = True
        else:
            self.currentIsBest = False

        if self.iteration > 10000:
            self.model.gfactor = 0.001
        history = {k:v.numpy() for k, v in history.items()}
        return history

    def save_checkpoint(self, checkpoint_dir):
        if self.currentIsBest:
            self.model.autoencoder.save(os.path.join(checkpoint_dir,"AE.h5"))
            self.model.save_image(self.val_output, self.iteration, logdir=checkpoint_dir)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        
        pass 

    # def reset_config(self, new_config):
    #     self.model = resAAE(**new_config)
    #     self.val_loss = 0.01
    #     self.currentIsBest=False

#===============set up dataset================

datapath = r'/uctgan/data/udelCT'
file_reference = r'./data/udelCT/File_reference.csv'
train_set = np.load(open(os.path.join(datapath, "trainset.npy"), "rb"))
val_set = np.load(open(os.path.join(datapath, "valset.npy"), "rb"))

def stopper(trial_id, result):
    conditions = [
        result["AE_loss"]  < 0.001, 
        result["AE_loss"]  > 1 and result["training_iteration"] > 250 ,
        result["AE_loss"]  > 0.2 and result["training_iteration"] > 500 ,
    ]
    return reduce(lambda x,y: x or y, conditions)

class MyCallback(Callback):
    def on_trial_complete(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['metric']}")

asha_scheduler = AsyncHyperBandScheduler(
    time_attr='training_iteration',
    max_t=20000,
    grace_period=20000,
    reduction_factor=3,
    brackets=1)

analysis = tune.run(
    tune.with_parameters(AAETrainable, batch_size=16, epochs=5000, data={"train":train_set, "val":val_set}),
    name="resAAETF-onoff",
    metric="val_loss",
    mode="min",
    scheduler=asha_scheduler,
    local_dir="/uctgan/data/ray_results",
    verbose=1,
    num_samples=1,
    resources_per_trial={
        "cpu": 3,
        "gpu": 1
    },
    stop=stopper, 
    config = {
        "g_loss_factor":0.00000001,
        "hidden": grid_search([(8,8,16,16),(8,16,16,32),(8,16,32,64)]),
        "hidden_D":(8, 16, 16, 32),
        "optAE_lr" : 0.001, 
        "optAE_beta" : 0.9,
        "optD_lr" : PiecewiseConstantDecay(boundaries = [10000,], values = [0.00000001, 0.0001]), 
        "optD_beta" : 0.5,
        "img_shape": (48, 96, 96, 1), 
        "loss_AE": mixedMSE(filter=gaussianFilter3D(sigma=1, kernel_size=3), alpha=0.5, mode="add"), 
        "loss_GD": BinaryCrossentropy(from_logits=False),
        "last_decoder_act": relu,
        "acc": MeanSquaredError(),
        "d_dropout": 0.1,
        "output_slices": slice(None)
        },
    keep_checkpoints_num = 10,
    checkpoint_score_attr="min-val_loss",
    checkpoint_freq=1,
    checkpoint_at_end=True,
    fail_fast=True
)
print("Best hyperparameters found were: ", analysis.best_config)