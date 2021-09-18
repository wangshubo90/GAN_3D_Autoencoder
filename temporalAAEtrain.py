import os, glob, json, pickle
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf

import SimpleITK as sitk 
import numpy as np
from model.temporalAAEv2 import temporalAAEv2

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

logdir=r"..\Gan_log\Nongan_AAE-TF-no-noise"
config = pickle.load(open(os.path.join(logdir, "config.pkl"), "rb"))
model_checkpoints = glob.glob(os.path.join(logdir, "*.h5"))
best_model = natsorted(model_checkpoints)[-1]

tAAE = temporalAAEv2(
    AAE_config=config, 
    AAE_checkpoint=best_model,
    lstm_layers=[32,32]
    )

fileRef = r"../training/seq_file_ref.csv"
dataSrcDir = r"../data"

def readDatasetGenerator(file, dataSrcDir, subset="train", batch_size=3, randseed=42, seq_length=4):
    """
    subset : choose from "train", "validate", "evaluate"
    """
    df = pd.read_csv(file)
    df = df[df["subset"]==subset]
    # df = {i:df[df["seq_length"]==i].reset_index() for i in range(2,6)}
    random.seed(randseed)
    def generator():
        sidx=0
        while True:
            # length = random.randint(2,5)
            # subdf = df[length].sample(n=batch_size, random_state=randseed+sidx)
            subdf = df.sample(n=batch_size, random_state=randseed+sidx)
            file_seq_list =  subdf.file_names.apply(lambda x: [i.strip("'") for i in x.strip("[]").split(", ")]).to_list()
            target_list = subdf.tar_img.to_list()
            
            xdata = np.zeros(shape=(len(file_seq_list), seq_length, 48, 96, 96, 1), dtype=np.float32)
            ydata = np.zeros(shape=(len(file_seq_list), 48, 96, 96, 1), dtype=np.float32)
            for idx, (seq, tar) in enumerate(zip(file_seq_list, target_list)):
                if len(seq) > 4:
                    seq = seq[-4:] 
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

train_set = readDatasetGenerator(fileRef, dataSrcDir, "train")
val_set = readDatasetGenerator(fileRef, dataSrcDir, "validate")
logdir=r"../Gan_log/temporalAAE-first-run"
summary = tAAE.train(train_set, val_set, 10, logdir=logdir, logstart=1, logimage=3, slices=[slice(None), 36])
# x, y = next(traingen)
# val_x, val_y = next(valgen)

# a = np.zeros(shape=(3,4,48,96,96,1))
# b = np.zeros((3, 48,96,96,1), np.float32)
# _=tAAE.__train_step__(a, b, a, b, 0.01)
# _=tAAE.__train_step__(x, y, val_x, val_y, 0.01)




