import numpy as np
import pandas as pd 
import tensorflow as tf 
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import os

def normalize(img):
    if type(img) == sitk.Image:
        img = sitk.GetArrayFromImage(img)

    img = img[:,:,:,np.newaxis].astype(np.float32) / 255.
    return img

def data_reader(file_ls, shuffle_size, batch_size, seed=42):
    train_img, test_img = train_test_split(file_ls, test_size=0.3, random_state=seed)
    val_img, evl_img = train_test_split(test_img, test_size=0.5, random_state=seed)

    def dataGenerator(file_ls):
        def generator():
            for file in file_ls:
                img = sitk.ReadImage(file)
                img = img[2:98,2:98,:]
                img = normalize(img)

            yield img
        return generator

    def datasetReader(generator, shuffle_size, batch_size):
        dataset = tf.data.Dataset.from_generator(generator,
            output_types = tf.float32,
            output_shapes = tf.TensorShape((48, 96, 96, 1)))
        dataset = dataset.repeat().batch(batch_size).shuffle(shuffle_size)
    
        return dataset

    trainds = datasetReader(dataGenerator(train_img), shuffle_size, batch_size)
    valds = datasetReader(dataGenerator(val_img), shuffle_size, batch_size)
    evlds = datasetReader(dataGenerator(evl_img), shuffle_size, batch_size)

    return trainds, valds, evlds

def data_reader_np(trainset, shuffle_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(trainset)
    dataset = dataset.repeat().batch(batch_size).shuffle(shuffle_size)

    return dataset

if __name__=="__main__":
    import glob
    datapath = r'/uctgan/data/udelCT'
    img_ls = glob.glob(os.path.join(datapath, "*.nii*"))
    trainds, _, _ =data_reader(img_ls, 32, 16)

    train_set = np.zeros(shape=[len(img_ls), 48, 96, 96, 1])
    idx = 0

    from tqdm import tqdm
    for file in tqdm(img_ls):
        img = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img)
        img = img[:,2:98,2:98,np.newaxis].astype(np.float32) / 255.
        train_set[idx] = img
        idx += 1

    train_set = data_reader_np(train_set, 48, 16)
    a, b = train_set.take(2)
    print(a.shape)