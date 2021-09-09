import numpy as np

def addNoise(image, mean, std):
    backgroundmask = np.nonzero(image == 0)
    image[backgroundmask] = np.random.normal(mean, std, size=backgroundmask[0].shape[0])
    return image