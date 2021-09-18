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
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

# logdir=r"..\Gan_log\Nongan_AAE-TF-no-noise"
# config = pickle.load(open(os.path.join(logdir, "config.pkl"), "rb"))
# model_checkpoints = glob.glob(os.path.join(logdir, "*.h5"))
# best_model = natsorted(model_checkpoints)[-1]

config = r"D:\gitrepos\data\ray_results\resAAETF-dynamicGAN-no-flatten\AAETrainable_dca02_00001_1_hidden=(8, 16, 16, 32)_2021-09-11_15-45-48\params-py3.pkl"
config = pickle.load(open(config, "rb"))
best_model=r"D:\gitrepos\data\ray_results\resAAETF-dynamicGAN-no-flatten\AAETrainable_dca02_00001_1_hidden=(8, 16, 16, 32)_2021-09-11_15-45-48\checkpoint_018193\AE-18193.h5"
best_D=r"D:\gitrepos\data\ray_results\resAAETF-dynamicGAN-no-flatten\AAETrainable_dca02_00001_1_hidden=(8, 16, 16, 32)_2021-09-11_15-45-48\checkpoint_018193\D-18193.h5"

tAAE = temporalAAE(
    AAE_config=config, 
    AAE_checkpoint=best_model,
    D_checkpoint=best_D,
    lstm_layers=[32]
    )

fileRef = r"../data/seq_file_ref.csv"
dataSrcDir = r"../data/ct/data"

def readDatasetGenerator(file, dataSrcDir, subset="train", batch_size=4, randseed=42):
    """
    subset : choose from "train", "validate", "evaluate"
    """
    df = pd.read_csv(file)
    df = df[df["subset"]==subset]
    df = {i:df[df["seq_length"]==i].reset_index() for i in range(2,6)}
    random.seed(randseed)
    def generator():
        sidx=0
        while True:
            length = random.randint(2,5)
            subdf = df[length].sample(n=batch_size, random_state=randseed+sidx)
            file_seq_list =  subdf.file_names.apply(lambda x: [i.strip("'") for i in x.strip("[]").split(", ")]).to_list()
            target_list = subdf.tar_img.to_list()
            
            xdata = np.zeros(shape=(len(file_seq_list), length, 48, 96, 96, 1), dtype=np.float32)
            ydata = np.zeros(shape=(len(file_seq_list), 48, 96, 96, 1), dtype=np.float32)
            for idx, (seq, tar) in enumerate(zip(file_seq_list, target_list)):
                for step, file in enumerate(seq):
                    img = sitk.ReadImage(os.path.join(dataSrcDir, file))
                    img = sitk.GetArrayFromImage(img)
                    img = img[:, 2:98, 2:98, np.newaxis] / 255
                    xdata[idx, step, ...] = img
                img = sitk.ReadImage(os.path.join(dataSrcDir, tar))
                img = sitk.GetArrayFromImage(img)
                img = img[:, 2:98, 2:98, np.newaxis] / 255

                ydata[idx] = img
            sidx+=1
            yield xdata, ydata
    return generator()

traingen = readDatasetGenerator(fileRef, dataSrcDir, "train")
valgen = readDatasetGenerator(fileRef, dataSrcDir, "validate")

tf.keras.backend.set_value(tAAE.optimizer_autoencoder.learning_rate, 0.01)
# tAAE.AAE.discriminator.load_weights(best_D)
# tAAE.discriminator = tAAE.AAE.discriminator

tAAE.train(traingen, valgen, 3, 10000, logdir=r"../data/experiments/temporalAAE-First", logstart=5, logimage=4)



