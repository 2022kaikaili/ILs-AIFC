import os
import platform
import numpy as np
import pandas as pd

from scaler import Standardization
from utils.featurizer import get_canonical_smiles

# 'Si', 'Al', 'Mg', 'Zn', 'Cr', 'Na', 'K', 'Pb'
ATOM_REMOVE = ['Ca']


def import_extra_dataset(params):
    DATASET_NAME = params['Dataset']
    path = os.path.join(
            './/datasets',
            DATASET_NAME + '.csv')
    data_from = os.path.realpath(path)
    # Lad the data
    if DATASET_NAME == 'AF':
        df = pd.read_csv(data_from, sep=';', encoding='utf-8')
    else:
        df = pd.read_csv(data_from, sep=',', encoding='utf-8')

    df['SMILES'] = get_canonical_smiles(df['SMILES'])
    Scaling = Standardization(df['VALUE'])
    df['VALUE'] = Scaling.Scaler(df['VALUE'])
    return df, Scaling
    # return df


def import_extra_dataset_TP(params):
    DATASET_NAME = params['Dataset']
    path = os.path.join(
            './/datasets',
            DATASET_NAME + '.csv')
    data_from = os.path.realpath(path)
    # Lad the data
    if DATASET_NAME == 'AF':
        df = pd.read_csv(data_from, sep=';', encoding='utf-8')
    else:
        df = pd.read_csv(data_from, sep=',', encoding='utf-8')

    df['SMILES'] = get_canonical_smiles(df['SMILES'])
    Scaling_value = Standardization(df['VALUE'])
    df['VALUE'] = Scaling_value.Scaler(df['VALUE'])
    Scaling_T = Standardization(df['TEMPERATURE'])
    df['TEMPERATURE'] = Scaling_T.Scaler(df['TEMPERATURE'])
    Scaling_P = Standardization(df['PRESSURE'])
    df['PRESSURE'] = Scaling_P.Scaler(df['PRESSURE'])

    return df, Scaling_value, Scaling_T, Scaling_P



