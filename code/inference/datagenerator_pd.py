from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import numpy as np
#np.random.seed(1)
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size, dim, shuffle,filename, column):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.filename = filename
        self.column = column
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)  # sites

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load based on the ID from csv filter by ID
            dataset = pd.read_csv(self.filename)
            dataset = dataset[dataset['Subject']==ID]
            #print(ID)
            path = dataset['Path'].values
            #print(path)
            itk_img = sitk.ReadImage(path)
            np_img = sitk.GetArrayFromImage(itk_img)
            X[i,] = np.float32(np_img.reshape(self.dim[0], self.dim[1], self.dim[2], 1))
            y[i,] = dataset[self.column].values 
        

        return X, y # This line will take care of outputing the inputs for training and the labels
