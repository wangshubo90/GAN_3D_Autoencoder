import os
os.environ["PYTHONPATH"] = r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\GAN_autoencoder"
import tensorflow as tf
import SimpleITK as sitk 
import numpy as np
from model.resAAE import resAAE

#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '3'

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

import matplotlib as plt

fig, ax = plt.subplots(2, 2, figsize=(16,12))

ax[0].plot(range(2999), history["AE_loss"], label="AE_loss")
ax[0].title("AE_loss")

fig, ax = plt.subplots(2,2)

ax[0, 0].plot(range(2999), history["AE_loss"], label="AE_loss")
ax[0, 0].set_title("AE_loss")

ax[0, 1].plot(range(2999), history["G_loss"], label="G_loss")
ax[0, 1].set_title("G_loss")

ax[1, 0].plot(range(2999), history["D_loss"], label="D_loss")
ax[1, 0].set_title("D_loss")

plt.show()