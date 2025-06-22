# -*- coding: utf-8 -*-
# @Time    : 2025/6/22 02:56
# @Author  : Kai Kai
# @Site    :
# @File    : atom_featurizer.py
# @Software: PyCharm

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EState import EState
import torch


ATOM_VOCAB = ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl',
               'Br', 'I', 'In', 'As', 'Ga', 'Fe', 'Au', 'Re', 'Sb']   # 19


def one_of_k_encoding_unk(x, allowable_set):
    # one-hot features converter
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def chirality_type(x, allowable_set):
    # atom's chirality type
    if x.HasProp('_CIPCode'):
        return one_of_k_encoding_unk(str(x.GetProp('_CIPCode')), allowable_set)
    else:
        return [0, 0]


def node_feature(atom):
    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']) +
        [atom.GetIsAromatic()] +
        [atom.HasProp('_ChiralityPossible')] +
        chirality_type(atom, ['R', 'S']) +
        [atom.GetFormalCharge()]
    )


def node_feature_mol_level(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        charges.append(float(atom.GetProp('_GasteigerCharge')))

    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    mol_log = []
    mol_mr = []
    for x, y in crippen:
        mol_log.append(x)
        mol_mr.append(y)

    asa = rdMolDescriptors._CalcLabuteASAContribs(mol)
    lasa = [x for x in asa[0]]
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    estate = EState.EStateIndices(mol)

    return np.column_stack([charges] +
                           [mol_log] +
                           [mol_mr] +
                           [lasa] +
                           [list(tpsa)] +
                           [estate.tolist()])


def classic_atom_featurizer(mol):
    num_atoms = mol.GetNumAtoms()  #
    atom_list = [mol.GetAtomWithIdx(i) for i in range(num_atoms)]
    node_feats = np.asarray([node_feature(atom) for atom in atom_list])

    return torch.tensor(node_feats)


def extended_atom_featurizer(mol):
    num_atoms = mol.GetNumAtoms()
    atom_list = [mol.GetAtomWithIdx(i) for i in range(num_atoms)]
    node_feats = np.asarray([node_feature(atom) for atom in atom_list])
    x_node_feats = node_feature_mol_level(mol)
    ext_node_feats = np.concatenate((node_feats, x_node_feats), axis=1)
    return torch.tensor(ext_node_feats)


# # test
# mol = Chem.MolFromSmiles('CCCCCCCCCCCCn1cc[n+](c1)CC[C@@H](CCC=C(C)C)C')
# feature = classic_atom_featurizer(mol)
# print(feature.shape)  # [27,46]