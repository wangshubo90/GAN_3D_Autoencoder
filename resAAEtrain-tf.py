from __future__ import absolute_import
import os, glob, json, pickle
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf

import SimpleITK as sitk 
import numpy as np
from model.resAAETF import resAAE
from tqdm import tqdm
from utils.losses import *
from functools import partial
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import tanh
from tensorflow.keras.losses import MeanSquaredError

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

config={
    "g_loss_factor":0.1,
    "hidden_D":(16, 32, 64),
    "optD_lr":0.0001,
    "optD_beta":0.5,
    "optAE_lr":0.0005,
    "optAE_beta":0.9,
    "img_shape": (48, 96, 96, 1), 
    "encoded_dim": 16, 
    "loss_AE": mixedGradeintError, 
    "loss_GD": MeanSquaredError(),
    "acc": MeanSquaredError(),
    "hidden": (8, 16, 32, 64),
    "output_slices": slice(None),
    "batch_size": 16,
    "epochs": 5000
}
#[slice(None), slice(None,15), slice(2,62), slice(2,62), slice(None)]
model = resAAE(**config)
logdir = r"..\data\experiments\test_AAE-TF"
os.makedirs(logdir, exist_ok=True)
json.dump({k:str(v) for k, v in config.items()}, open(os.path.join(logdir, "config.json"), "w"))
pickle.dump(config, open(os.path.join(logdir, "config.pkl"), "wb"))
#===============set up dataset================
def read_data(file_ls):
    dataset = np.zeros(shape=[len(file_ls), 48, 96, 96, 1], dtype=np.float32)
    
    for idx, file in enumerate(file_ls):
        img = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img)
        img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
        dataset[idx] = img
    return dataset   
datapath = r'..\data\ct\Data'
file_reference = r'..\data\ct\File_reference.csv'
# datapath = r"/uctgan/data/udelCT"
# file_reference = r"/uctgan/data/Gan_training/File_reference.csv"
img_ls = glob.glob(os.path.join(datapath, "*.nii.gz"))
seed = 42
random.seed(seed)
np.random.seed(42)
random.shuffle(img_ls)

train_img, test_img = train_test_split(img_ls, test_size=0.3, random_state=seed)
val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

train_set = read_data(train_img) 
val_set = read_data(val_img)

# test = model.train_step(train_set, val_set, 16)
history = model.train(train_set, val_set, 16, 10000, logdir=logdir, logstart=500)


