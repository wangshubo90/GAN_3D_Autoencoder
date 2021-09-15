import os, glob, json, pickle
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf

import SimpleITK as sitk 
import numpy as np
from model.temporaAAE import temporalAAE

from tqdm import tqdm
from utils.losses import *
from functools import partial
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import tanh
from natsort import natsorted

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

logdir=r"..\Gan_log\Nongan_AAE-TF-no-noise"
config = pickle.load(open(os.path.join(logdir, "config.pkl"), "rb"))
model_checkpoints = glob.glob(os.path.join(logdir, "*.h5"))
best_model = natsorted(model_checkpoints)[-1]

tAAE = temporalAAE(
    AAE_config=config, 
    AAE_checkpoint=best_model,
    lstm_layers=[32,32]
    )

a = tf.ones((3,48,96,96,1))
a = tAAE.encoder(a)
a = tAAE.temporalModel(a)
a = tAAE.decoder(a)
print(a.shape)

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
