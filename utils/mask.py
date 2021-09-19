import tensorflow as tf

def compute_mask(inputTensor, mask_value, reduce_axes, keepdims=False):
    mask = tf.keras.backend.any(
        tf.not_equal(inputTensor, mask_value), axis=reduce_axes, keepdims=keepdims)
    return mask