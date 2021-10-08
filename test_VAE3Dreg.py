import tensorflow as tf
from model.VAEmrs import VAEmrs, build_model
import numpy as np
import os

shape = (60,60,15,6)
model = VAEmrs(shape, 6)


x = np.zeros(shape=(1, *shape), dtype=np.float32)
y = np.ones(shape=(1, 6,6,8), dtype=np.float32)

r = model.train_step(model.fetch_batch(1,1))
r[0]