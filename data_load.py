import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import image
import PIL
from imgaug import augmenters as iaa


def load_data(data_dir, data_aug=False):

    #files = os.listdir(data_dir)
    files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    
    reduced_height = 240
    reduced_width = 135

    #X = np.zeros((len(files), 960, 540, 3))
    #Y = np.zeros((len(files), 960, 540, 1))
    
    X = np.zeros((len(files), reduced_height, reduced_width, 3))
    Y = np.zeros((len(files), reduced_height, reduced_width, 1))
    X_aug = np.zeros((len(files), reduced_height, reduced_width, 3))
    Y_aug = np.zeros((len(files), reduced_height, reduced_width, 1))


    for i, img in enumerate(files):
        if img == ".DS_Store" or img == "._.DS_Store":
            continue


        path = os.path.join(data_dir, img)

        # Load image as array (not to be used)
        image_array = image.imread(path)

        # Load image as PIL object
        image_obj = PIL.Image.open(path)

        # Crop images
        input_img = image_obj.crop((0,0,540,960))
        mask_img = image_obj.crop((540,0,1080,960))
        
        # Resize images (here half size)
        input_img = input_img.resize((reduced_width, reduced_height), PIL.Image.ANTIALIAS)
        mask_img = mask_img.resize((reduced_width, reduced_height), PIL.Image.ANTIALIAS)

        # Convert to arrays
        input_data = np.asarray(input_img)
        target_data = np.asarray(mask_img)[:, :, :1]

        # Store data
        X[i], X_aug[i] = input_data, input_data
        Y[i], Y_aug[i] = target_data, target_data


    X = np.reshape(X, [len(files), 3, reduced_height, reduced_width])
    Y = np.reshape(Y, [len(files), 1, reduced_height, reduced_width])

    bg = (Y < 200).astype(int)
    hand = (Y >= 200).astype(int)
    Y_onehot = np.concatenate((bg, hand), axis=1)
    Y = np.argmax(Y_onehot, axis=1)

    if data_aug:

    	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    	seq = iaa.Sequential([
    	iaa.Crop(px=(10, 40)), # crop images from each side by 10 to 40px (randomly chosen)
    	iaa.Fliplr(0.5), # horizontally flip 50% of the images
    	#sometimes(iaa.ElasticTransformation(alpha=50, sigma=5)),
    	#iaa.Dropout([0.05, 0.2]),
    	sometimes(iaa.GaussianBlur(sigma=(0, 5))),
    	#iaa.Sharpen((0.0, 1.0)),
    	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    	iaa.Affine(
    		#scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    		#translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    		rotate=(-90, 90),
    		#shear=(-8, 8)
    		)
    	], random_order=True)

    	X_aug, Y_aug = seq(images=X_aug.astype('uint8'), segmentation_maps=Y_aug.astype('uint8'))

    	X_aug = np.reshape(X_aug, [len(files), 3, reduced_height, reduced_width])
    	Y_aug = np.reshape(Y_aug, [len(files), 1, reduced_height, reduced_width])

    	bg_aug = (Y_aug < 200).astype(int)
    	hand_aug = (Y_aug >= 200).astype(int)
    	Y_onehot_aug = np.concatenate((bg_aug, hand_aug), axis=1)
    	Y_aug = np.argmax(Y_onehot_aug, axis=1)

    	return X, Y, X_aug, Y_aug

    return X, Y
