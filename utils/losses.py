import tensorflow as tf
from tensorflow import keras
import numpy as np
try:
    from utils import filters
except ModuleNotFoundError:
    import filters

def focalSoftMSE(y_true, y_pred, alpha=0.1, beta=10.0):

    squareError = tf.math.squared_difference(y_true, y_pred)
    softplus = tf.math.log(1+alpha*tf.math.exp(squareError * beta + 3))
    loss = squareError * softplus

    return tf.reduce_mean(loss)

def focalImageLoss(y_true, y_pred, threshold):
    return

def focalMSE(y_true, y_pred, alpha=0.1, gamma=2.0):
    """
    Description: focal MSE loss
    """
    mse = keras.losses.mse(y_true, y_pred)
    loss =alpha* tf.pow(1-y_true, gamma) * mse[:,:,:,:,tf.newaxis]

    return tf.reduce_sum(loss)

def meanGradientError(y_true, y_pred):
    """
    Description: mean gradient error
    """
    mge = keras.losses.mse(filters.sobelFilter3D(y_true), filters.sobelFilter3D(y_pred))

    return tf.reduce_mean(mge)


def mixedGradeintError(y_true, y_pred, alpha=0.001):
    """
    Description: Mixed gradient error
    """
    mge = keras.losses.mse(filters.sobelFilter3D(y_true), filters.laplacianFilter3D(y_pred))
    mse = keras.losses.mse(y_true, y_pred)

    return tf.reduce_mean(alpha * mge + (1-alpha)*mse)

class mixedMSE():
    def __init__(self, filter, mode="add", alpha=0.001, **kwargs):
        self.filter = filter
        self.alpha = alpha
        self.kwargs = kwargs
        self.mode = mode

    def __call__(self, y_true, y_pred):
        tobemix = keras.losses.mse(self.filter(y_true, **self.kwargs), self.filter(y_pred, **self.kwargs))
        mse = keras.losses.mse(y_true, y_pred)
        if self.mode=="add":
            loss = tf.reduce_mean(self.alpha * tobemix + mse)
        elif self.mode=="blend":
            loss = tf.reduce_mean(self.alpha * tobemix + (1-self.alpha)*mse)
        
        
        return loss

def cDice(y_true, y_pred):
    by = tf.cast(tf.where(y_true>0, 1, 0), tf.float32)
    intersect = tf.reduce_sum(by*y_pred)
    if intersect >0:
        c = tf.reduce_sum(by*y_pred)/tf.reduce_sum(tf.cast(tf.where(y_pred>0,1,0), tf.float32)*by)
    else:
        c = 1
    cdice = 2*intersect / (c*tf.reduce_sum(y_pred)+tf.reduce_sum(by))
    return cdice

class mixedDiceMSE():
    def __init__(self, filter, mode="add", alpha=0.001, **kwargs):
        self.filter = filter
        self.alpha = alpha
        self.kwargs = kwargs
        self.mode = mode
        
    def cDice(self, y_true, y_pred):
        # by = tf.cast(tf.where(y_true>0, 1, 0), tf.float32)
        by = y_true
        intersect = tf.reduce_sum(by*y_pred)
        # if intersect >0:
        #     c = tf.reduce_sum(by*y_pred)/tf.reduce_sum(tf.cast(tf.where(y_pred>0,1,0), tf.float32)*by)
        # else:
        #     c = 1
        cdice = 2*intersect / (tf.reduce_sum(y_pred**2)+tf.reduce_sum(by**2))
        return cdice
        
    def __call__(self, y_true, y_pred):
        tobemix = 1 - self.cDice(y_true, y_pred)
        mse = keras.losses.mse(y_true, y_pred)
        if self.mode=="add":
            loss = tf.reduce_mean(self.alpha * tobemix + mse)
        elif self.mode=="blend":
            loss = tf.reduce_mean(self.alpha * tobemix + (1-self.alpha)*mse)
        return loss

class focalSoftDice():
    def __init__(self, filter, mode="add", alpha=0.001, **kwargs):
        self.filter = filter
        self.alpha = alpha
        self.kwargs = kwargs
        self.mode = mode

    def __call__(self, y_true, y_pred):
        y_true_filtered = self.filter(y_true, **self.kwargs)
        dice = 2 * tf.reduce_sum(y_true_filtered*y_pred) / tf.reduce_sum(y_true_filtered+y_pred)

        tobemix = keras.losses.mse(self.filter(y_true, **self.kwargs), self.filter(y_pred, **self.kwargs))
        mse = keras.losses.mse(y_true, y_pred)
        if self.mode=="add":
            loss = tf.reduce_mean(self.alpha * tobemix + mse)
        elif self.mode=="blend":
            loss = tf.reduce_mean(self.alpha * tobemix + (1-self.alpha)*mse)
        
        
        return loss
    
if __name__=="__main__":
    
    import SimpleITK as sitk

    #image = sitk.GetArrayFromImage(sitk.ReadImage(r"/uCTGan/data/unitTest/test_t1_brain.nii.gz"))
    image = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\Data\236LT_w1.nii.gz"))
    #lossfn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    lossfn = focalSoftMSE
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