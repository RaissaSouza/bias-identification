from pickle import FALSE
from datagenerator_pd import DataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
random.seed(1)
import argparse

params = {'batch_size': 1,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160,
        'column': "Sex_bin"
        }

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_test', type=str, help='filename to infer')
parser.add_argument('-model_name', type=str, help='model to infer')
parser.add_argument('-output', type=str, help='output name')
args = parser.parse_args()
import csv


def calculate_metrics(y_test,y_pred,fn):
    line = []
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print(cm)
    cm_df = pd.DataFrame(cm,
            index = ['F','M'], 
            columns = ['F','M'])
      #Plotting the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, cmap="Blues", annot=True,fmt='.2f', vmin=0, vmax=1.0, center=0.5,cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(fn,bbox_inches='tight')
    plt.show()
    
    ac=accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    line.append(ac)
    line.append(sens)
    line.append(spec)


  
    return line
    


fn_test = args.fn_test
test = pd.read_csv(fn_test)
IDs_list=test['Subject'].to_numpy()
test_IDs=IDs_list
test_generator=DataGenerator(test_IDs, 1, (
    params['imagex'], params['imagey'], params['imagez']), False, fn_test, params['column'])


# reload the best performing model for the evaluation
model=tf.keras.models.load_model(args.model_name)
# make sure that model weights are non-trainable
model.trainable=False

name=args.output


#Make predictions and save the results
y_test = test[params['column']].values 
y_pred=model.predict(test_generator)
y_pred = (y_pred>=0.5)
y_pred = y_pred.astype(int)
pred = (np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
df = pd.DataFrame(pred)
df.to_csv('pred_'+name+'.csv', index=FALSE)
metrics = calculate_metrics(y_test, y_pred,'test_cm_'+name+'.png')
df = pd.DataFrame(metrics)
df.to_csv('metrics_'+name+'.csv', index=FALSE)



