import os, glob, json, pickle
import tensorflow as tf
import numpy as np
from model.resAAETF import resAAE   
import matplotlib.pyplot as plt
import SimpleITK as sitk 
from sklearn.model_selection import train_test_split
from natsort import natsorted
# checkpoint_dir = "/uctgan/data/ray_results/AAE_uct_test2/AAETrainable_16ccf_00002_2_optD_lr=3.3526e-06,optG_lr=1.6763e-05_2021-08-25_00-47-57"
# checkpoint = "checkpoint_002479"
# param = "params.json"
# config = json.load(open(os.path.join(checkpoint_dir, param)))
logdir=r"D:\gitrepos\data\experiments\test_AAE-TF"
config = pickle.load(open(os.path.join(logdir, "config.pkl"), "rb"))
model_checkpoints = glob.glob(os.path.join(logdir, "*.h5"))
best_model = natsorted(model_checkpoints)[-1]

model = resAAE(**config)
model.autoencoder.load_weights(best_model)
# checkpoint_restore = tf.train.Checkpoint(
#     model=model.autoencoder,
#     optimizers = model.autoencoder.optimizer,
# )
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint_restore.restore(latest).assert_consumed()

# datapath = r'/uctgan/data/udelCT'
# file_reference = r'./data/udelCT/File_reference.csv'
# datapath = r'..\Data'
# file_reference = r'..\Training\File_reference.csv'
datapath = r'..\data\ct\Data'
train_set = np.load(open(os.path.join(datapath, "trainset.npy"), "rb"))
val_set = np.load(open(os.path.join(datapath, "valset.npy"), "rb"))
# file_reference = r'..\data\ct\File_reference.csv'
# seed=42
# img_ls = glob.glob(os.path.join(datapath, "*.nii*"))
# train_img, test_img = train_test_split(img_ls, test_size=0.3, random_state=seed)
# val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

for i in range(-11, -8):
    image = val_set[i]
    gen_image=np.squeeze(model.autoencoder.predict(image[np.newaxis,...]) * 255)
    image = np.squeeze(image)*255

    fig2, ax = plt.subplots(2,3, figsize=(16,9))
    ax[0,0].imshow(image[35,:,:],cmap="gray")
    ax[1,0].imshow(gen_image[35,:,:],cmap="gray")
    ax[0,1].imshow(image[::-1,50,:],cmap="gray")
    ax[1,1].imshow(gen_image[::-1,50,:],cmap="gray")
    ax[0,2].imshow(image[::-1,:,60],cmap="gray")
    ax[1,2].imshow(gen_image[::-1,:,60],cmap="gray")