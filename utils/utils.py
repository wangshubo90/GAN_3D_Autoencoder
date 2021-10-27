import os, glob, shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imsave

def show_image(output, logdir=".", logimage=8, slices=None, save_to_file=None):
    image = np.squeeze(output[0].numpy())[:logimage]
    valimage = np.squeeze(output[1])[:logimage]
    shape = image.shape
    if slices:
        image = image[slices]
        valimage = valimage[slices]
    else:
        image = image[:, shape[1]//2, ...]
        valimage = valimage[:, shape[1]//2, ...]
    image = np.concatenate([image.reshape((-1, shape[-1])), valimage.reshape((-1, shape[-1]))], axis=1)

    fig = plt.figure()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.close()

    if save_to_file:
        imsave(save_to_file, image, check_contrast=False)

def cleanRayTrial(trialdir):
    for i in glob.glob(trialdir):
        if os.path.isdir(i):
            if len(glob.glob(os.path.join(i, "*.h5"))) == 0:
                shutil.rmtree(i)
