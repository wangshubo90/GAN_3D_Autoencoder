import os, glob, json
import tensorflow as tf
import numpy as np
from model.AAE import AAE   
import matplotlib.pyplot as plt
import SimpleITK as sitk 
from sklearn.model_selection import train_test_split
checkpoint_dir = "/uctgan/data/ray_results/AAE_uct_test2/AAETrainable_16ccf_00002_2_optD_lr=3.3526e-06,optG_lr=1.6763e-05_2021-08-25_00-47-57"
checkpoint = "checkpoint_002479"
param = "params.json"
config = json.load(open(os.path.join(checkpoint_dir, param)))

model = AAE(**config)
model.autoencoder.load_weights("/uctgan/data/ray_results/AAE_uct_test/AAETrainable_70a1a_00000_0_optD_lr=7.5963e-07,optG_lr=9.8752e-06_2021-08-24_04-19-14/checkpoint_000950/AE.h5")
# checkpoint_restore = tf.train.Checkpoint(
#     model=model.autoencoder,
#     optimizers = model.autoencoder.optimizer,
# )
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint_restore.restore(latest).assert_consumed()

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
train_img, test_img = train_test_split(img_ls, test_size=0.3, random_state=seed)
val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

image = sitk.GetArrayFromImage(sitk.ReadImage(val_img[-9]))
image= image[:, 2:98, 2:98].reshape((1,48,96,96,1))
gen_image=np.squeeze(model.autoencoder.predict(image.astype(np.float32)/255)) * 255
image = np.squeeze(image)

fig2, ax = plt.subplots(2,3, figsize=(16,9))
ax[0,0].imshow(image[35,:,:],cmap="Greys")
ax[1,0].imshow(gen_image[35,:,:],cmap="Greys")
ax[0,1].imshow(image[::-1,50,:],cmap="Greys")
ax[1,1].imshow(gen_image[::-1,50,:],cmap="Greys")
ax[0,2].imshow(image[::-1,:,60],cmap="Greys")
ax[1,2].imshow(gen_image[::-1,:,60],cmap="Greys")