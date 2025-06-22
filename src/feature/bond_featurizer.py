# -*- coding: utf-8 -*-
# @Time    : 2025/6/22
# @Author  : Kai kai,
# @Site    : 
# @File    : bond_featurizer.py
# @Software: PyCharm

from rdkit import Chem
import numpy as np
from rdkit.Chem.rdchem import BondType
import torch


def edge_feature(bond):
    bt = bond.GetBondType()
    stereo = bond.GetStereo()
    fbond = np.asarray([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        stereo == Chem.rdchem.BondStereo.STEREONONE,
        stereo == Chem.rdchem.BondStereo.STEREOANY,
        stereo == Chem.rdchem.BondStereo.STEREOZ,
        stereo == Chem.rdchem.BondStereo.STEREOE,
        stereo == Chem.rdchem.BondStereo.STEREOCIS,
        stereo == Chem.rdchem.BondStereo.STEREOTRANS
    ])
    return fbond

# commonly-used bond featurizer
def classic_bond_featurizer(mol):
    num_bonds = mol.GetNumBonds()
    edge_feats = np.zeros([2 * num_bonds, 12])
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        edge_feats[2 * i, :] = edge_feature(bond)
        edge_feats[2 * i + 1, :] = edge_feature(bond)
    return torch.tensor(edge_feats)

#test
# mol = Chem.MolFromSmiles('CCCCCCCCCCCCn1cc[n+](c1)CC[C@@H](CCC=C(C)C)C')
# features = classic_bond_featurizer(mol)
# print(features.shape) # [54,12]