# An explainable ionic liquid property model by coupling ionic fragment contribution with graph neural network
***


## Background
***
This code is the basis of our work submitted to *AIChE Journal*, aiming to 
integrate *a-priori* knowledge, i.e. ionic fragment contribution theory, with graph neural networks and attention mechanism, and bring more insights in the prediction of properties of ionic liquids. This is an **alpha version** of the models used to generate the results published in:

[An explainable ionic liquid property model by coupling ionic fragment contribution with graph neural network]


## Prerequisites and dependencies
***
The code requires the following packages and modules:\
\
bayesian_optimization==1.4.3\
dgl==1.1.2+cu116\
numpy==1.24.3\
openpyxl==3.1.2\
pandas==2.0.3\
prettytable==3.9.0\
rdkit==2023.9.6\
scikit_learn==1.3.2\
seaborn==0.13.0\
torch==1.13.1\
tqdm==4.66.1

## Usage
***
### Fragmentation and Cache Files
The relevant scheme for fragmentation and their corresponding SMARTs are kept in './datasets/My_fragments.csv'.

Meanwhile the folders storing cache files and error logs would be created automatically in the root directory after the first time of scripts running.
### Training and SplittingSeed Tuning 

For properties independent of temperature and pressure, use the following command to train the ionic fragment (IF)-based model and tune the splitting seed:
```commandline
$ python Seed_IFC.py 
```
For properties dependent on temperature and pressure, use the following command:
```commandline
$ python Seed_IFC_PT.py 
```
The results of every attempt containing metrics on three folds are automatically printed in './output/'.
### Optimization
To perform Bayesian optimization on an IF-based model for properties independent of temperature and pressure, use:
```commandline
$ python new_optimization_IFC.py
```
To perform Bayesian optimization on an IF-based model for properties dependent on temperature and pressure, use:
```commandline
$ python new_optimization_IFC_PT.py
```
### Generate Ensemble Models
To generate ensemble models with random initializations for properties independent of temperature and pressure, run:
```commandline
$ python ensembling_IFC.py
```
To generate ensemble models for properties dependent on temperature and pressure, run:
```commandline
$ python ensembling_IFC_PT.py
```
Note that the default ensemble size is 100. And the trained models would be saved in folders named by the model names under './library/Ensembles/', which would be created when these scripts operating.

After the ensemble models generated, this script need to be carried out to print the predictions of every compounds in dataset together with their latent representations.
```commandline
$ python ensemble_IFC_compound.py
$ python ensemble_IFC_compound_PT.py
```
All these outputs are available under the folders where models are saved in.
## Contribution
***
Xiangping Zhang ([xpzhang@ipe.ac.cn](xpzhang@ipe.ac.cn))

Kaikai Li ([likaikai20@ipe.ac.cn](likaikai20@ipe.ac.cn))

## Licence
***
Check the licence file

