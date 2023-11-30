import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import csv
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, Nadam
from tensorflow.keras.regularizers import l1_l2, l1, l2
#from tensorflow_addons.layers import GroupNormalization, WeightNormalization
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
import argparse



#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-model_name', type=str, help='model name to save')
parser.add_argument('-encoder', type=str, help='model to load')
args = parser.parse_args()


params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160,
        'column': "Sex_bin"
        }


def load_model():
    model=tf.keras.models.load_model(args.encoder)
    layer_output = model.get_layer('dense1').output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    intermediate_model.summary()
    intermediate_model.trainable=False
    return intermediate_model

def add_final_layer(intermediate_model):
    dense_layer=Dense(units=1, activation='sigmoid',name="dense2")(intermediate_model.output)
    model = tf.keras.models.Model(inputs=intermediate_model.input, outputs=dense_layer)
    model.summary()
    opt = Adam(lr=0.001, decay=0.003)
    metr = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metr)
    return model

    
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


fn_train = args.fn_train
train = pd.read_csv(fn_train)
IDs_list = train['Subject'].to_numpy()
train_IDs = IDs_list
training_generator = DataGenerator(train_IDs,params['batch_size'],(params['imagex'], params['imagey'], params['imagez']),True,fn_train,params['column'])


fn_val = args.fn_test
val = pd.read_csv(fn_val)
IDs_list = val['Subject'].to_numpy()
val_IDs = IDs_list
val_generator = DataGenerator(val_IDs, params['batch_size'],(params['imagex'], params['imagey'], params['imagez']),True,fn_val, params['column'])

intermidiate_model = load_model()
model = add_final_layer(intermidiate_model)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(args.model_name+".h5", monitor='val_loss', verbose=2,
                                                         save_best_only=True, include_optimizer=True,
                                                         save_weights_only=False, mode='auto',
                                                         save_freq='epoch')

history = model.fit(training_generator, epochs=1000, validation_data=val_generator,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),checkpoint_callback], verbose=2)

history_dict = history.history


from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
