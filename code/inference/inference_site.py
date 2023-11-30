from pickle import FALSE
from datagenerator import DataGenerator
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
import csv
import argparse

params = {'batch_size': 1,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160,
        'column': 'Site_2',
        'number_class': 41
        }
parser = argparse.ArgumentParser()
parser.add_argument('-fn_test', type=str, help='filename to infer')
parser.add_argument('-model_name', type=str, help='model to infer')
parser.add_argument('-output', type=str, help='output name')
args = parser.parse_args()


def calculate_metrics(y_test,y_pred,fn):
    line = []
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print(cm)
    cm_df = pd.DataFrame(cm,
                    index = ['BIOCOG','C-BIG','HAMBURG','HMC','Japan','JGH','MUC','Neurocon','OASIS','CALGARY','PLS','PPMI_10','PPMI_12','PPMI_13','PPMI_14','PPMI_15','PPMI_16','PPMI_17','PPMI_18','PPMI_19','PPMI_20','PPMI_21','PPMI_22','PPMI_23','PPMI_25','PPMI_26','PPMI_27','PPMI_28','PPMI_29','PPMI_30','PPMI_51','PPMI_52','PPMI_53','PPMI_55','PPMI_59','RUH','SALD','SBK','Taowu','UKBB','UOA'], 
                    columns = ['BIOCOG','C-BIG','HAMBURG','HMC','Japan','JGH','MUC','Neurocon','OASIS','CALGARY','PLS','PPMI_10','PPMI_12','PPMI_13','PPMI_14','PPMI_15','PPMI_16','PPMI_17','PPMI_18','PPMI_19','PPMI_20','PPMI_21','PPMI_22','PPMI_23','PPMI_25','PPMI_26','PPMI_27','PPMI_28','PPMI_29','PPMI_30','PPMI_51','PPMI_52','PPMI_53','PPMI_55','PPMI_59','RUH','SALD','SBK','Taowu','UKBB','UOA'])
    ac=accuracy_score(y_test, y_pred)
    print(ac)
    #Plotting the confusion matrix
    plt.figure(figsize=(10,8))
    g1=sns.heatmap(cm_df, cmap="Blues", annot=False,fmt='.2f', vmin=0, vmax=1.0, center=0.5,cbar=True)
    plt.title('T1-weighted')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(fn,format='png', dpi=1200,bbox_inches='tight')
    plt.show()
    line.append(ac)
    return line
    


fn_test = args.fn_test
test = pd.read_csv(fn_test)
IDs_list=test['Subject'].to_numpy()
test_IDs=IDs_list
test_generator=DataGenerator(test_IDs, 1, (
    params['imagex'], params['imagey'], params['imagez']), False, fn_test, params['column'],params['number_class'])

# reload the best performing model for the evaluation
model=tf.keras.models.load_model(args.model_name)
# make sure that model weights are non-trainable
model.trainable=False


name=args.output


#Make predictions and save the results
y_test = test[params['column']].values 
y_pred=model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
pred = (np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
df = pd.DataFrame(pred)
df.to_csv('pred_'+name+'.csv', index=FALSE)
metrics = calculate_metrics(y_test, y_pred,'test_cm_'+name+'.png')
df = pd.DataFrame(metrics)
df.to_csv('metrics_'+name+'.csv', index=FALSE)







