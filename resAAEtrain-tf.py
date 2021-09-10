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
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from utils.noise import addNoise
from utils.filters import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

config={
    "g_loss_factor":0.000000001,
    "hidden_D":(16, 32, 64, 128),
    "optD_lr":0.000000001,
    "optD_beta":0.5,
    "optAE_lr":0.001,
    "optAE_beta":0.9,
    "img_shape": (48, 96, 96, 1), 
    "encoded_dim": 16, 
    "loss_AE": mixedMSE(filter=gaussianFilter3D(sigma=1, kernel_size=3), alpha=0.5, mode="add"), 
    "loss_GD": BinaryCrossentropy(from_logits=False),
    "last_decoder_act": tanh,
    "acc": MeanSquaredError(),
    "hidden": (8, 16, 16, 32),
    "d_dropout": 0.1,
    "output_slices": slice(None),
    "batch_size": 16,
    "epochs": 5000
}
#[slice(None), slice(None,15), slice(2,62), slice(2,62), slice(None)]
model = resAAE(**config)
logdir = r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\Gan_log\Nongan_AAE-TF-no-noise-8-16-16-32"
os.makedirs(logdir, exist_ok=True)
json.dump({k:str(v) for k, v in config.items()}, open(os.path.join(logdir, "config.json"), "w"))
pickle.dump(config, open(os.path.join(logdir, "config.pkl"), "wb"))
#===============set up dataset================
# def read_data(file_ls):
#     dataset = np.zeros(shape=[len(file_ls), 48, 96, 96, 1], dtype=np.float32)
    
#     for idx, file in enumerate(file_ls):
#         img = sitk.ReadImage(file)
#         img = sitk.GetArrayFromImage(img)
#         # img = addNoise(img, 20, 12)
#         img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
#         dataset[idx] = img
#     return dataset   
datapath = r'..\Data'
file_reference = r'..\Training\File_reference.csv'
# # datapath = r"/uctgan/data/udelCT"
# # file_reference = r"/uctgan/data/Gan_training/File_reference.csv"
# img_ls = glob.glob(os.path.join(datapath, "*.nii.gz"))
# seed = 42
# random.seed(seed)
# np.random.seed(42)
# random.shuffle(img_ls)

# train_img, test_img = train_test_split(img_ls, test_size=0.3, random_state=seed)
# val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

# train_set = read_data(train_img) 
# val_set = read_data(val_img)
# evl_set = read_data(evl_img)

# np.save(open(os.path.join(datapath, "trainset.npy"), "wb"), train_set)
# np.save(open(os.path.join(datapath, "valset.npy"), "wb"), val_set)
# np.save(open(os.path.join(datapath, "evlset.npy"), "wb"), evl_set)

# import matplotlib.pyplot as plt
# plt.imshow(np.squeeze(train_set[0])[0,...], cmap="gray")

train_set = np.load(open(os.path.join(datapath, "trainset.npy"), "rb"))
val_set = np.load(open(os.path.join(datapath, "valset.npy"), "rb"))
evl_set = np.load(open(os.path.join(datapath, "evlset.npy"), "rb"))

# test = model.train_step(train_set, val_set, 16)
history = model.train(train_set, val_set, 16, 10000, logdir=logdir, logstart=500)


