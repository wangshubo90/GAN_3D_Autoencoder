import numpy as np
import pandas as pd 
import tensorflow as tf 
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import os

def RotateAugmentation(img):
    '''
    Description: rotate the input 3D image along z-axis by random angles 
    Args:
        img: sitk.Image, dimension is 3
    Return: sitk.Image
    '''
    img.SetOrigin((0,0,0))
    img.SetSpacing((1.0, 1.0, 1.0))
    dim = img.GetSize()
    translation = [np.random.randint(0, 5), np.random.randint(0, 5), 0]
    angle = np.float64(np.random.randint(0, 360) / 180 * np.pi)
    rotation = [np.float64(0), np.float64(0), angle] # x-axis, y-axis, z-axis angles
    center = [(d-1) // 2 for d in dim]
    transformation = sitk.Euler3DTransform()    
    transformation.SetCenter(center)            # SetCenter takes three args
    transformation.SetRotation(*rotation)         # SetRotation takes three args
    transformation.SetTranslation(translation)
    aug_img = sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), 
            sitk.Cast(img[0:96, 0:96, :], 
            sitk.sitkFloat32), transformation, 
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
            )
    return aug_img

def IdentityAugmentation(img):
    return img[2:98, 2:98, :]

def DataGenerator(file_ls, label_ls, augmentation = None, example_n = 2, special_multiplier = 1):
    def normalize(img):
        if type(img) == sitk.Image:
            img = sitk.GetArrayFromImage(img)

        img = img[:,:,:,np.newaxis].astype(np.float32) / 255.
        return img

    def generator():
        for f, y in zip(file_ls, label_ls):
            y = np.array([y]).astype(np.float32)
            img = sitk.ReadImage(os.path.join(datapath, f))
            orig_img = img[2:98, 2:98, :]
            if augmentation is not None:
                if y == 0:
                    imgls = [normalize(orig_img)]
                    N = example_n
                    for _ in range(N):
                        imgls.append(normalize(augmentation(img)))

                    for img in imgls:
                        yield img

                elif y > 0 :
                    imgls = [normalize(orig_img)]
                    N = example_n * special_multiplier
                    for _ in range(N):
                        imgls.append(normalize(augmentation(img)))

                    for img in imgls:
                        yield img
        
            else:
                yield normalize(orig_img)

    return generator

def DatasetReader(file_ls, label_ls, shuffle_size, batch_size, augmentation = None, example_n = 2, special_multiplier = 1):
    generator = DataGenerator(file_ls, label_ls, augmentation=augmentation, 
            example_n = example_n, special_multiplier = special_multiplier
    )
    dataset = tf.data.Dataset.from_generator(generator,
        output_types = tf.float32,
        output_shapes = tf.TensorShape((48, 96, 96, 1))
    )
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_size).batch(batch_size)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

def Reader(file_ref, data_path, augmentation = RotateAugmentation):
    np.random.seed(42)
    datapath = data_path  #path to data dir
    file_df = pd.read_csv(file_ref)

    file_ls = file_df['File name'].to_numpy()
    label = file_df['Label'].to_numpy()
    label = label.astype('int64')

    train_img, test_img, train_l, test_l = train_test_split(
            file_ls, label, test_size=0.3, random_state=42
    )
    #del imgs
    # split test data into validation and evaluation evenly
    val_img, evl_img, val_l, evl_l = train_test_split(
            test_img, test_l, test_size = 0.5, random_state=42
    )
    '''
    print(train_img.shape)
    print(val_img.shape)
    print(evl_img.shape)

    print(sum(train_l >= 1) / train_l.shape[0])
    print(sum(val_l >= 1) / val_l.shape[0])
    print(sum(evl_l >= 1) / evl_l.shape[0])
    '''
    BATCH_SIZE = 36

    train_set = DatasetReader(train_img, train_l, 640, BATCH_SIZE,
            augmentation = RotateAugmentation, example_n = 1, special_multiplier = 2
    )
    val_set = DatasetReader(val_img, val_l, 180, BATCH_SIZE) #, augmentation = IdentityAugmentation, example_n = 1, special_multiplier = 1
    evl_set = DatasetReader(evl_img, evl_l, 180, BATCH_SIZE)
    return train_set, val_set, evl_set