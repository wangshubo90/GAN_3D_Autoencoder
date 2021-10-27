from tensorflow.python.ops.gen_math_ops import AddN
from model.temporalAAEv2 import temporalAAEv2
import os, glob, json, pickle
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import relu
import SimpleITK as sitk 
import numpy as np
from model.temporalAAEv3 import temporalAAEv3

from tqdm import tqdm
from utils.losses import *
from utils.filters import *
from utils.noise import addNoise
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

# config = r"D:\gitrepos\data\ray_results\resAAETF-dynamicGAN-no-flatten\AAETrainable_dca02_00001_1_hidden=(8, 16, 16, 32)_2021-09-11_15-45-48\params-py3.pkl"
# config = pickle.load(open(config, "rb"))
config = {
        "g_loss_factor":0.001,
        "hidden": (8,8,16,16),
        "hidden_D":(8, 16, 16, 32),
        "optAE_lr" : 0.001, 
        "optAE_beta" : 0.9,
        "optD_lr" : PiecewiseConstantDecay(boundaries = [10000,], values = [0.00000001, 0.0001]), 
        "optD_beta" : 0.9,
        "img_shape": (48, 96, 96, 1), 
        "loss_AE": mixedMSE(filter=gaussianFilter3D(sigma=1, kernel_size=3), alpha=0.5, mode="add"), 
        "loss_GD": BinaryCrossentropy(from_logits=False),
        "last_decoder_act": relu,
        "acc": MeanSquaredError(),
        "d_dropout": 0.1,
        "output_slices": slice(None)
        }

best_model=r"D:\gitrepos\data\ray_results\resAAETF-noflatten-gf0.01-beta0.6-latest\AAETrainable_0266a_00001_1_g_loss_factor=0.001,hidden=(8, 8, 16, 16)_2021-09-21_00-33-03\checkpoint_029001\AE-29001.h5"
# best_D=r"D:\gitrepos\data\ray_results\resAAETF-noflatten-gf0.01-beta0.6-latest\AAETrainable_0266a_00001_1_g_loss_factor=0.001,hidden=(8, 8, 16, 16)_2021-09-21_00-33-03\checkpoint_029001\D-29001.h5"

tAAE = temporalAAEv2(
    AAE_config=config, 
    AAE_checkpoint=best_model,
    D_checkpoint=None,
    lstm_layers=[128,128]
    )

tAAE.optimizer_autoencoder=Adam(learning_rate=0.001)
tAAE.optimizer_discriminator=Adam(PiecewiseConstantDecay(boundaries = [500,2500], values = [0.0001, 0.0005, 0.001]))
tAAE.gfactor=0.01

fileRef = r"../data/seq_file_ref.csv"
dataSrcDir = r"../data/ct/data"

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
                    img = addNoise(img, 20+sidx % 3, 10+sidx % 2)
                    xdata[idx, step, ...] = img
                img = sitk.ReadImage(os.path.join(dataSrcDir, tar))
                img = sitk.GetArrayFromImage(img)
                img = img[:, 2:98, 2:98, np.newaxis] / 255

                ydata[idx] = img
            sidx+=1
            yield xdata, ydata
    return generator()

train_set = readDatasetGenerator(fileRef, dataSrcDir, "train", batch_size=16)
val_set = readDatasetGenerator(fileRef, dataSrcDir, "validate", batch_size=16)
logdir=r"..\data\experiments\temporalAAEv2-16-GAN-noise"
summary = tAAE.train(train_set, val_set, 10000, logdir=logdir, logstart=100, logimage=6, slices=[slice(None), 36])
# x, y = next(traingen)
# val_x, val_y = next(valgen)

# a = np.zeros(shape=(3,4,48,96,96,1))
# b = np.zeros((3, 48,96,96,1), np.float32)
# _=tAAE.__train_step__(a, b, a, b, 0.01)
# _=tAAE.__train_step__(x, y, val_x, val_y, 0.01)



