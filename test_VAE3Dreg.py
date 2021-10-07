import tensorflow as tf
from model.VAE3Dseg import VAE3Dseg
import numpy as np
import os

model = VAE3Dseg((16,128,128,4),5)

shape = (16,128,128,4)
x = np.zeros(shape=(1, *shape), dtype=np.float32)
y = np.ones(shape=(1, 16,128,128,5), dtype=np.float32)

r = model.train_step(1,1)