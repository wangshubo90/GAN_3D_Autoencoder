import tensorflow as tf
from tensorflow import keras
import numpy as np

def focalImageLoss(y_true, y_pred, threshold):
    return

def focalMSE(y_true, y_pred, alpha, gamma):
    mse = keras.losses.mse(y_true, y_pred)

    loss =alpha* y_true * mse

    return loss

def GradientLoss(y_true, y_pred):
    return

if __name__=="__main__":
    
    import SimpleITK as sitk
    image = sitk.GetArrayFromImage(sitk.ReadImage(r"/uCTGan/data/unitTest/test_t1_brain.nii.gz"))
    tfimage = tf.convert_to_tensor(image, dtype=tf.float32)    
    tfimage = tfimage[tf.newaxis,:,:,:,tf.newaxis]
    #tfimage = tf.reshape(tfimage, shape=(1,*tfimage.shape,1))
    tfzero = tf.zeros_like(tfimage)

