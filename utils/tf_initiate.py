import os
import logging
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['AUTOGRAPH_VERBOSITY'] = '2'
import tensorflow as tf
tf.autograph.set_verbosity(2)
# tf.get_logger().setLevel(logging.ERROR)

import horovod.tensorflow as hvd

#======================= Set up Horovod ======================
# comment out this chunk of code if you train with 1 gpu
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')