# -*- coding: utf-8 -*-
# @Time    : 2025/6/23 14:31
# @Author  : Kai kai
# @Site    :
# @File    : mol_featurizer.py
# @Software: PyCharm

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import numpy as np
import torch


def global_feature(mol):
    num_atom = mol.GetNumAtoms()
    num_bond = mol.GetNumBonds()
    mole_weight = Descriptors.MolWt(mol)
    num_aliph_ring = rdMolDescriptors.CalcNumAliphaticRings(mol)
    num_aroma_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    return np.asarray([num_atom,
                       mole_weight,
                       num_aliph_ring,
                       num_aroma_ring])


def classic_mol_featurizer(mol):
    num_atom = mol.GetNumAtoms()
    global_feats = np.asarray([global_feature(mol) for _ in range(num_atom)])
    return torch.tensor(global_feats)


# test
# mol = Chem.MolFromSmiles('CCCCCCCCCCCCn1cc[n+](c1)CC[C@@H](CCC=C(C)C)C')
# features = classic_mol_featurizer(mol)
# print(features.shape) # [27,4]
