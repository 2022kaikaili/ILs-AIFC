import math
import numpy as np
import platform
import os
import io
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem, rdCoordGen, rdGeometry
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from collections import defaultdict
import itertools
import json


def visualize_frag_weights(path, smiles_list, targets_list, tag_list, predictions_list, attentions_list_array, atom_mask_list): #相关调用的代码在csv_dataset中
    save_path = os.path.realpath('.//library/' + path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dict = {'SMILE':[], 'Target':[], 'Prediction':[], 'Residual':[],  'Tag':[]}
    for i, smi in enumerate(smiles_list):
        dict['SMILE'].append(smi)
        dict['Target'].append(targets_list[i])
        dict['Prediction'].append(predictions_list[i])
        dict['Tag'].append(tag_list[i])
        attention = attentions_list_array[i]
        atom_mask = atom_mask_list[i]
        if not math.isnan(targets_list[i]):
            dict['Residual'].append(abs(predictions_list[i] - targets_list[i]))
        else:
            dict['Residual'].append(np.nan)
        visualize_frag_weights_single(save_path, smi, i, attention, atom_mask)
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(save_path + '/' +
              'config.csv', index=False)


def print_frag_attentions(path, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list):
    save_path = os.path.realpath('.//library/' + path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mol_dict = {}
    for i, smi in enumerate(smiles_list):
        mol_dict[str(smi)] = {}
        assert atom_mask_list[i].shape[0] == len(frag_flag_list[i]), 'Numbers of fragments are unmatched.'
        #max_att = max(attentions_list_array[i][:, time_step])
        #min_att = min(attentions_list_array[i][:, time_step])
        for j in range(atom_mask_list[i].shape[0]):
            mol_dict[str(smi)]['frag_' + str(j)] = {}
            mol_dict[str(smi)]['frag_' + str(j)]['frag_name'] = frag_flag_list[i][j]
            x = attentions_list_array[i][j]
            mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float(x)
            #if max_att != min_att:
                #mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float((x - min_att) / (max_att - min_att))
            #else:
                #mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float(x)

    with open(save_path + '/' + 'fragments_info.json', 'w', newline='\n') as f:
        str_ = json.dumps(mol_dict, indent=1)
        f.write(str_)





