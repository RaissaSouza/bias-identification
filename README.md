# Shortcut Learning - Biases Identification
<div align="center">

</div>

<p align="center">
<img src="workflow.png?raw=true">
</p>


Implementation for potential biases identification in a Parkinson's disease classifier that is under review by the IEEE Journal of Biomedical and Health Informatics (J-BHI): "[Identifying biases in a multicenter MRI database for  Parkinson’s disease classification: Is the disease classifier a secret site classifier?] (https://doi.org/10.1109/JBHI.2024.3352513).

Our code here is based on the investigation of a Parkinson's disease classification model encode three sources of biases: sex, site, and scanner.

If you find our framework, code, or paper useful to your research, please cite us!
```
@article{Souza2024,
   author = {Raissa Souza and Anthony Winder and Emma A. M. Stanley and Vibujithan Vigneshwaran and Milton Camacho and Richard Camicioli and Oury Monchi and Matthias Wilms and Nils D. Forkert},
   doi = {10.1109/JBHI.2024.3352513},
   issn = {2168-2194},
   issue = {4},
   journal = {IEEE Journal of Biomedical and Health Informatics},
   month = {4},
   pages = {2047-2054},
   title = {Identifying Biases in a Multicenter MRI Database for Parkinson's Disease Classification: Is the Disease Classifier a Secret Site Classifier?},
   volume = {28},
   url = {https://ieeexplore.ieee.org/document/10388228/},
   year = {2024},
}

```
```
Souza, R., Winder, A., Stanley, E. A. M., Vigneshwaran, V., Camacho, M., Camicioli, R., Monchi, O., Wilms, M., & Forkert, N. D. (2024). Identifying Biases in a Multicenter MRI Database for Parkinson’s Disease Classification: Is the Disease Classifier a Secret Site Classifier? IEEE Journal of Biomedical and Health Informatics, 28(4), 2047–2054. https://doi.org/10.1109/JBHI.2024.3352513
```

### Abstract 
Sharing multicenter imaging datasets can be advantageous to increase data diversity and size but may lead to spurious correlations between site-related biological and non-biological image features and target labels, which machine learning (ML) models may exploit as shortcuts. To date, studies analyzing how and if deep learning models may use such effects as a shortcut are scarce. Thus, the aim of this work was to investigate if site-related effects are encoded in the feature space of an established deep learning model designed for Parkinson's disease (PD) classification based on T1-weighted MRI datasets. Therefore, all layers of the PD classifier were frozen, except for the last layer of the network, which was replaced by a linear layer that was exclusively re-trained to predict three potential bias types (biological sex, scanner type, and originating site). Our findings based on a large database consisting of 1880 MRI scans collected across 41 centers show that the feature space of the established PD model (74% accuracy) can be used to classify sex (75% accuracy), scanner type (79% accuracy), and site location (71% accuracy) with high accuracies despite this information never being explicitly provided to the PD model during original training. Overall, the results of this study suggest that trained image-based classifiers may use unwanted shortcuts that are not meaningful for the actual clinical task at hand. This finding may explain why many image-based deep learning models do not perform well when applied to data from centers not contributing to the training set.  

## PD classifier
We use the state-of-the-art simple fully convolutional network (SFCN) (doi: 10.1016/J.MEDIA.2020.101871) as our deep learning architecture. The Adam optimizer with an initial learning rate of 0.001, a decay rate of 0.003, and batch size 5 was used during training. The best model (lowest binary cross entropy testing loss) was saved for evaluation based on early stopping with patience of 10 epochs. 
The code used is in: 
```bash
├── code/pd_training
│   ├── sfcn_pd.py

```
To run this code, you will need to change the params variable to match the size of your image data and to read the correct column for your lables. After you update the file you can save and run:
```
python sfcn_pd.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -model_name ./best_model_pd
```

## Shortcut/bias identification
To identify possible shortcuts in the trained model, all layers of the pre-trained PD classifier model were frozen, except for the final layer, which was replaced with a customized linear layer designed to classify the specific biases of interest: sex (n=2), site (n=41), and scanner (n=19).
For sex classification, we employed a binary output layer with a sigmoid activation function, while for site and scanner type classification, multi-class output layers with softmax activation functions were used. For all cases, the Adam optimizer and early stopping were used as for the PD  model.
The code used for **SEX** exploration is in: 
```bash
├── code/sex_training
│   ├── sfcn_sex.py

```
To run this code, you will need to change the params variable to match the size of your image data and to read the correct column for your lables. After you update the file you can save and run:
```
python sfcn_sex.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -model_name ./best_model_sex -encoder ./best_model_pd.h5
```
* `encoder` is the pre-trained PD model

The code used for **SCANNER** exploration is in: 
```bash
├── code/scanner_training
│   ├── sfcn_scanner.py

```
To run this code, you will need to change the params variable to match the size of your image data, to read the correct column for your lables, and the number of classes in your dataset. After you update the file you can save and run:
```
python sfcn_scanner.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -model_name ./best_model_scanner -encoder ./best_model_pd.h5
```
* `encoder` is the pre-trained PD model
  
The code used for **SITE** exploration is in: 
```bash
├── code/site_training
│   ├── sfcn_site.py

```
To run this code, you will need to change the params variable to match the size of your image data, to read the correct column for your lables, and the number of classes in your dataset. After you update the file you can save and run:
```
python sfcn_scanner.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -model_name ./best_model_site -encoder ./best_model_pd.h5
```
* `encoder` is the pre-trained PD model


## Evaluation
The code used for evaluation is in: 
```bash
├── code/inference
│   ├── inference_pd.py
│   ├── inference_sex.py
│   ├── inference_site.py
│   ├── inference_scanner.py
```

## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.


## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=Bias-exploration).
