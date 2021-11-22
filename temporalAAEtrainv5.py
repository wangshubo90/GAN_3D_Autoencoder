import os, glob, json, pickle, shutil, sys
#================ Environment variables ================
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
from model.temporalAAEv5 import temporalAAEv5
from model import temporalAAEv5 as model_code
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], "GPU")
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

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


# logdir=r"..\Gan_log\Nongan_AAE-TF-no-noise"
# config = pickle.load(open(os.path.join(logdir, "config.pkl"), "rb"))
# model_checkpoints = glob.glob(os.path.join(logdir, "*.h5"))
# best_model = natsorted(model_checkpoints)[-1]

# config = r"D:\gitrepos\data\ray_results\resAAETF-dynamicGAN-no-flatten\AAETrainable_dca02_00001_1_hidden=(8, 16, 16, 32)_2021-09-11_15-45-48\params-py3.pkl"
# config = pickle.load(open(config, "rb"))
config = {
        "g_loss_factor":0.001,
        "hidden": (16,32,64,128),
        "hidden_D":(16, 32, 32, 64),
        "optAE_lr" : 0.001, 
        "optAE_beta" : 0.9,
        "optD_lr" : PiecewiseConstantDecay(boundaries = [10000,], values = [0.00000001, 0.0001]), 
        "optD_beta" : 0.9,
        "img_shape": (48, 96, 96, 1), 
        "loss_AE": mixedDiceMSE(filter=gaussianFilter3D(sigma=1, kernel_size=3), alpha=1, mode="blend"), 
        "loss_GD": BinaryCrossentropy(from_logits=False),
        "last_decoder_act": relu,
        "acc": MeanSquaredError(),
        "d_dropout": 0.1,
        "output_slices": slice(None)
        }


tAAE = temporalAAEv5(
    AAE_config=config, 
    AAE_checkpoint=None,
    D_checkpoint=None,
    lstm_layers=[128,128],
    loss_function_temporal=mixedDiceMSE(filter=gaussianFilter3D(sigma=1, kernel_size=3), alpha=1, mode="blend")
    )

# test_x = tf.zeros(shape=(4,4,48,96,96,1))
# test_y = tf.zeros(shape=(4,48,96,96,1))

tAAE.optimizer_autoencoder=Adam(learning_rate=0.001)
tAAE.optimizer_discriminator=Adam(PiecewiseConstantDecay(boundaries = [500,2500], values = [0.0001, 0.0005, 0.001]))
tAAE.gfactor=0.001

fileRef = r"/uctgan/data/udelCT/seq_file_ref_cleanup.csv"
dataSrcDir = r"/uctgan/data/udelCT"
np.random.seed(42)

def RandTransform(img_dim = (48,96,96), seed=42):
    translation = [
            np.round(np.random.normal(0, 3, 1)[0]), 
            np.round(np.random.normal(0, 3, 1)[0]), 
            np.round(np.random.normal(0, 2, 1)[0])]
    angle = np.float64(np.round(np.random.normal(0, 7, 1)[0]) / 180 * np.pi)
    rotation = [np.float64(0), np.float64(0), angle] # x-axis, y-axis, z-axis angles
    center = [(d-1) // 2 for d in img_dim]
    transformation = sitk.Euler3DTransform()    
    transformation.SetCenter(center)            # SetCenter takes three args
    transformation.SetRotation(*rotation)         # SetRotation takes three args
    transformation.SetTranslation(translation)
    
def RotateAugmentation(img, transform=None):
    '''
    Description: rotate the input 3D image along z-axis by random angles 
    Args:
        img: sitk.Image, dimension is 3
    Return: sitk.Image
    '''
    dim = img.GetSize()
    if not transform:
        translation = [
            np.round(np.random.normal(0, 3, 1)[0]), 
            np.round(np.random.normal(0, 3, 1)[0]), 
            np.round(np.random.normal(0, 2, 1)[0])]
        angle = np.float64(np.round(np.random.normal(0, 7, 1)[0]) / 180 * np.pi)
        rotation = [np.float64(0), np.float64(0), angle] # x-axis, y-axis, z-axis angles
        center = [(d-1) // 2 for d in dim]
        transformation = sitk.Euler3DTransform()    
        transformation.SetCenter(center)            # SetCenter takes three args
        transformation.SetRotation(*rotation)         # SetRotation takes three args
        transformation.SetTranslation(translation)
    # aug_ls = []
    
    img.SetOrigin((0,0,0))
    img.SetSpacing((1.0, 1.0, 1.0))

    aug_img = sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), 
            sitk.Cast(img, sitk.sitkFloat32), 
            transformation, 
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
            )
        # aug_ls.append(aug_img)
    return aug_img

def readDatasetGenerator(file, dataSrcDir, subset="train", batch_size=3, randseed=42, seq_length=4, augmentation=None):
    """
    subset : choose from "train", "validate", "evaluate"
    """
    df = pd.read_csv(file)
    if type(subset) is list:
        df = df[df["subset"].isin(subset)]
    else:
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
            if augmentation:
                aug_xdata = np.zeros(shape=(len(file_seq_list), seq_length, 48, 96, 96, 1), dtype=np.float32)
            xdata = np.zeros(shape=(len(file_seq_list), seq_length, 48, 96, 96, 1), dtype=np.float32)
            ydata = np.zeros(shape=(len(file_seq_list), 48, 96, 96, 1), dtype=np.float32)
            for idx, (seq, tar) in enumerate(zip(file_seq_list, target_list)):
                if len(seq) > 4:
                    seq = seq[-4:] 
                for step, file in enumerate(seq):
                    orgimg = sitk.ReadImage(os.path.join(dataSrcDir, file))
                    img = sitk.GetArrayFromImage(orgimg)
                    # img = addNoise(img, 20+sidx % 3, 10+sidx % 2)
                    img = img[:, 2:98, 2:98, np.newaxis] / 255
                    xdata[idx, step, ...] = img
                    
                    if augmentation:
                        augimg = augmentation(orgimg)
                        augimg = sitk.GetArrayFromImage(augimg)
                        augimg = augimg[:, 2:98, 2:98, np.newaxis] / 255
                        aug_xdata[idx, step, ...] = augimg
                    else:
                        aug_xdata = xdata
                    
                img = sitk.ReadImage(os.path.join(dataSrcDir, tar))
                img = sitk.GetArrayFromImage(img)
                # img = addNoise(img, 20+sidx % 3, 10+sidx % 2)
                img = img[:, 2:98, 2:98, np.newaxis] / 255

                ydata[idx] = img
            sidx+=1
            yield xdata, ydata, aug_xdata
    return generator()

train_set = readDatasetGenerator(fileRef, dataSrcDir, ["train","evaluate"], batch_size=4)
val_set = readDatasetGenerator(fileRef, dataSrcDir, "validate", batch_size=4)
logdir=r"data/experiments/temporalAAEv5-2nd-16-32-64-128-blend100-dice_squared-moredata-bidirect-AttBnm-temponly"
os.makedirs(logdir, exist_ok=True)
shutil.copy(model_code.__file__, logdir)
shutil.copy(__file__, logdir)
summary = tAAE.train(train_set, val_set, 10001, logdir=logdir, logfreq=2000, logstart=500, logimage=20, slices=[slice(None), 36])

#===== infer =====
# sys.path.append(logdir)
# from temporalAAEv5 import temporalAAEv5
# import matplotlib.pyplot as plt

# epoch = 10000
# model_weight = os.path.join(logdir, f"epoch-{epoch}", "tAAE_epoch_{}.h5".format(epoch))
# tAAE.temporalModel.load_weights(model_weight)

def inferGenerator(file, dataSrcDir, subset="eval"):
    """
    subset : choose from "train", "validate", "evaluate", or "visualize"
    """
    df = pd.read_csv(file)
    if subset == "visualize":
        subdf = df[df["abaqus"]==1]
    else:
        subdf = df[df["subset"]==subset]
    def generator():
        for idx, row in subdf.iterrows():
            file_seq_list =  [i.strip("'") for i in row.file_names.strip("[]").split(", ")]
            target_list = row.tar_img
            
            xdata = np.zeros(shape=(1, 4, 48, 96, 96, 1), dtype=np.float32)
            ydata = np.zeros(shape=(1, 48, 96, 96, 1), dtype=np.float32)
            seq = file_seq_list
            tar = target_list
            if len(seq) > 4:
                seq = seq[-4:] 
            for step, imgf in enumerate(seq):
                img = sitk.ReadImage(os.path.join(dataSrcDir, imgf))
                img = sitk.GetArrayFromImage(img)
                # img = addNoise(img, 20+sidx % 3, 10+sidx % 2)
                img = img[:, 2:98, 2:98, np.newaxis] / 255
                xdata[0, step, ...] = img
            img = sitk.ReadImage(os.path.join(dataSrcDir, tar))
            img = sitk.GetArrayFromImage(img)
                # img = addNoise(img, 20+sidx % 3, 10+sidx % 2)
            img = img[:, 2:98, 2:98, np.newaxis] / 255

            ydata[0] = img
            yield xdata, ydata, row.id, file_seq_list, tar
    return generator()
    
def plot_pred(dataset="train", save_results=False):
    eval_set = inferGenerator(fileRef, dataSrcDir, dataset)

    pred_eval = {}
    pred_eval_plot = {}
    
    if save_results:
        savedir = os.path.join(logdir, "visualize_pred")
        os.makedirs(savedir, exist_ok=True)
    for x, y, id, file_seq, tar in eval_set:
        y_pred = tAAE.temporalModel(x, training=False).numpy()
        x = np.squeeze(x)*255
        y_pred = y_pred[...,0]
        y = y[...,0]
        
        y = y*255
        y_pred = y_pred*255
        
        if save_results:
            sitk_y = sitk.GetImageFromArray(np.squeeze(y).astype(np.uint8))
            sitk_y_pred = sitk.GetImageFromArray(np.squeeze(y_pred).astype(np.uint8))
            sitk.WriteImage(sitk_y, os.path.join(savedir,tar))
            sitk.WriteImage(sitk_y_pred, os.path.join(savedir, "pred"+tar))
            
        evaluation = np.concatenate([x, y, y_pred], axis = 0)
        if id in pred_eval:
            pred_eval[id].append(evaluation)
            pred_eval_plot[id].append(evaluation[:, 32, ...].reshape((-1, 96)))
        else:
            pred_eval[id] = [evaluation]
            pred_eval_plot[id] = [evaluation[:, 32, ...].reshape((-1, 96))]

    for key in pred_eval.keys():
        figure = plt.figure()
        plt.imshow(np.concatenate(pred_eval_plot[key], axis=-1), cmap="gray")
        plt.axis("off")
        plt.title(key)
        plt.show()
        os.makedirs(os.path.join(logdir, "plots"+dataset), exist_ok=True)
        figure.savefig(os.path.join(logdir, "plots"+dataset, dataset+"_"+key+".png"), dpi=300)
        
    return pred_eval, pred_eval_plot