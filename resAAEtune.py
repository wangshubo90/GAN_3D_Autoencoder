import os 
import tensorflow as tf

#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

from model.resAAE import resAAE

config={
    "optG_lr":0.01,
    "optG_beta":0.01,
    "optD_lr":0.01,
    "optD_beta":0.01,
    "optAE_lr":0.01,
    "optAE_beta":0.01,
    "img_shape": (15, 60, 60, 1), 
    "encoded_dim": 16, 
    "loss": "mse", 
    "acc": "mse",
    "hidden": (16, 32, 64, 128),
    "output_slices": [slice(None), slice(None,15), slice(2,62), slice(2,62), slice(None)],
    "batch_size": 16,
    "epochs": 5000
}

model = resAAE(**config)