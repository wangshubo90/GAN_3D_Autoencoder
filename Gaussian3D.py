import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import scipy.stats as st


def gaussianMatrix(sigma, kernal):
    
    x = np.linspace(-sigma, sigma, kernal+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern3d = np.outer(kern1d, kern1d, kern1d)

    return kern3d/kern3d.sum()
    
def Gaussian3Dconv(sigma, kernal=None):
    if not kernal == None:
        kernal = sigma * 3
    return

if __name__=="__main__":

    print(gaussianMatrix(3, 9))