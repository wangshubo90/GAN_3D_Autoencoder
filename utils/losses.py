import tensorflow as tf
from tensorflow import keras
import numpy as np
try:
    from utils import filters
except ModuleNotFoundError:
    import filters

from filters import sobelFilter3D, gaussianFilter3D

def focalImageLoss(y_true, y_pred, threshold):
    return

def focalMSE(y_true, y_pred, alpha=1.0, gamma=5.0):
    mse = keras.losses.mse(y_true, y_pred)

    loss =alpha* tf.pow(1-y_true, gamma) * mse[:,:,:,:,tf.newaxis]

    return tf.reduce_mean(loss)

def meanGradientError(y_true, y_pred):
    mge = keras.losses.mse(sobelFilter3D(y_true), sobelFilter3D(y_pred))
    return tf.reduce_mean(mge)

def mixedGradeintError(y_true, y_pred, alpha=0.5):
    mge = keras.losses.mse(sobelFilter3D(y_true), sobelFilter3D(y_pred))
    mse = keras.losses.mse(y_true, y_pred)
    return tf.reduce_mean(alpha * mge + (1-alpha)*mse)
    
if __name__=="__main__":
    
    import SimpleITK as sitk

    image = sitk.GetArrayFromImage(sitk.ReadImage(r"/uCTGan/data/unitTest/test_t1_brain.nii.gz"))
    #image = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\Data\236LT_w1.nii.gz"))
    lossfn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    #lossfn = focalMSE
    tf.random.set_seed(42)


    tfimage = tf.convert_to_tensor(image, dtype=tf.float32) / tf.reduce_max(image)
    tfimage = tfimage[tf.newaxis,:,:,:,tf.newaxis]
    #tfimage = tf.reshape(tfimage, shape=(1,*tfimage.shape,1))
    tfzero = tf.zeros_like(tfimage)

    input = keras.layers.Input(shape =tfimage.shape[1:])
    convlayer = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1,)*3, padding="SAME", use_bias=False, kernel_initializer="he_normal")
    output = convlayer(input)
    minimodel = tf.keras.models.Model(input, output)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(1000):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.MeanAbsolutePercentageError()

        with tf.GradientTape() as tape:
            trueimg = filters.gaussianFilter3D(tfimage, 0.5, 3)
            predimg = minimodel(tfimage)
            loss = lossfn(trueimg, predimg)
        
        grads = tape.gradient(loss, minimodel.trainable_weights)
        optimizer.apply_gradients(zip(grads, minimodel.trainable_variables))

        epoch_loss_avg.update_state(loss)
        epoch_accuracy.update_state(trueimg, minimodel(tfimage))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 200 == 0:
            #print(loss)
            print(np.squeeze(minimodel.layers[-1].weights[0].numpy()))
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                    epoch_loss_avg.result(),
                                    epoch_accuracy.result()))
    
    print(tf.reduce_mean(
        keras.losses.mse(
            filters.gaussianFilter3D(tfimage, 0.5,3), minimodel(tfimage)
        )
    ))