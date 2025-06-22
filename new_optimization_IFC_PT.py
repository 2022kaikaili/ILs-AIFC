# -*- coding: utf-8 -*-
# @Time    : 2024/03/22 8:19
# @Author  : Kai kai
# @Site    :
# @File    : optimization.py
# @Software: PyCharm

from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.junctiontree_encoder import JT_SubGraph
from utils.Earlystopping import EarlyStopping
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import  collate_fraggraphs
import torch
from torch.utils.data import DataLoader
import os
import time
import pandas as pd
from utils.count_parameters import count_parameters
from utils.splitter import Splitter
from data.dataloading import import_extra_dataset_TP
from networks.AIFC import AIFCNet
from networks.IFCGAT import IFCGATNet
from utils.Set_Seed_Reproducibility import set_seed
from utils.piplines_TP import train_epoch_IFC_TP, evaluate_IFC_TP

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from bayes_opt import UtilityFunction

params = {}
net_params = {}
params['init_lr'] = 1e-3
params['min_lr'] = 1e-9
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.8
params['lr_schedule_patience'] = 20
params['earlystopping_patience'] = 50
params['max_epoch'] = 600

net_params['num_atom_type'] = 46
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 128
net_params['num_heads'] = 1
net_params['dropout'] = 0
net_params['depth'] = 2
net_params['layers'] = 2
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cuda'
dataset_list = ['viscosity']

splitting_list = [222]


def main(params, net_params):
    # model = IFCGATNet(net_params).to(device='cuda')
    model = AIFCNet(net_params).to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                           patience=params['lr_schedule_patience'], verbose=False)
    # early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint1' + params['Dataset'] + 'IFCGAT' + '.pt')
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint1' + params['Dataset'] + 'AIFC' + '.pt')


    n_param = count_parameters(model)
    for epoch in range(params['max_epoch']):
        model, epoch_train_loss, epoch_train_metrics = train_epoch_IFC_TP(model, optimizer, scaling, train_loader, n_param)
        epoch_val_loss, epoch_val_metrics = evaluate_IFC_TP(model, scaling, val_loader, n_param)


        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print('\n! LR equal to min LR set.')
            break

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    model = early_stopping.load_checkpoint(model)
    _, epoch_train_metrics = evaluate_IFC_TP(model, scaling, train_loader, n_param)
    _, epoch_val_metrics = evaluate_IFC_TP(model, scaling, val_loader, n_param)
    _, epoch_test_metrics = evaluate_IFC_TP(model, scaling, test_loader, n_param)
    _, epoch_raw_metrics = evaluate_IFC_TP(model, scaling, raw_loader, n_param)

    return epoch_train_metrics, epoch_val_metrics, epoch_test_metrics, epoch_raw_metrics


def Splitting_Main_MO(params, net_params):
    """Built-in function. Use the basic block of implementation in Multi-objective Bayesian Optimization.
    Parameters
    ----------
    params : dict
        Set of parameters for workflow.
    net_params : dict
        Set of parameters for architectures of models.

    Returns
    ----------
    -train_metrics.RMSE : float
        Optimization Objective-1, negative number of RMSE metric in training
    -val_metrics.RMSE : float
        Optimization Objective-2, negative number of RMSE metric in validation
    """
    train_metrics, val_metrics, test_metrics, all_metrics = main(params, net_params)
    return -val_metrics.RMSE


def func_to_be_opt_MO(hidden_dim, depth, layers, decay, dropout, init_lr, lr_reduce_factor):
    """Built-in function. Objective function in Single-objective Bayesian Optimization.
    Parameters
    ----------
    Returns
    ----------
    """
    net_params['hidden_dim'] = int(hidden_dim)
    net_params['layers'] = int(layers)
    net_params['depth'] = int(depth)
    # net_params['num_heads'] = int(num_heads)
    params['weight_decay'] = 10 ** (-int(decay))
    params['init_lr'] = 10 ** (-init_lr)
    params['lr_reduce_factor'] = 0.4 + 0.05 * int(lr_reduce_factor)
    net_params['dropout'] = dropout

    return Splitting_Main_MO(params, net_params)


set_seed(seed=1000)
torch.manual_seed(222)
for i in range(len(dataset_list)):
    params['Dataset'] = dataset_list[i]
    df, scaling, T, P = import_extra_dataset_TP(params)
    cache_file_path = os.path.realpath('./cache')
    if not os.path.exists(cache_file_path):
        os.mkdir(cache_file_path)
    cache_file = os.path.join(cache_file_path, params['Dataset'] + '_CCC')

    error_path = os.path.realpath('./error_log')
    if not os.path.exists(error_path):
        os.mkdir(error_path)
    error_log_path = os.path.join(error_path, '{}_{}'.format(params['Dataset'], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

    fragmentation = JT_SubGraph('My_fragments')
    net_params['frag_dim'] = fragmentation.frag_dim
    dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer, classic_mol_featurizer, cache_file, load=True
                                 , error_log=error_log_path, fragmentation=fragmentation)
    splitter = Splitter(dataset)

    rows = []
    file_path = os.path.realpath('./output')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    # save_file_path = os.path.join(file_path,
    #                               '{}_{}_{}_{}_{}'.format('SOPT_UCB_5', params['Dataset'], 'IFCGAT', splitting_list[i],
    #                                                       time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
    save_file_path = os.path.join(file_path,
                                  '{}_{}_{}_{}_{}'.format('SOPT_UCB_5', params['Dataset'], 'AIFC', splitting_list[i],
                                                          time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
    set_seed(seed=1000)
    train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=splitting_list[i], frac_train=0.8, frac_val=0.1)

    train_loader = DataLoader(train_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)
    raw_loader = DataLoader(raw_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)

    hpbounds = {'hidden_dim': (16, 256.99), 'depth': (1, 5.99), 'layers': (1, 5.99), 'decay': (0, 6.99), 'dropout': (0, 0.9), 'init_lr': (2, 5), 'lr_reduce_factor': (0, 10.99)}
    #bounds_transformer = SequentialDomainReductionTransformer()
    mutating_optimizer = BayesianOptimization(f=func_to_be_opt_MO, pbounds=hpbounds, verbose=1, random_state=22)
    utility = UtilityFunction(kind="ucb", kappa=5, xi=0.0)
    # mutating_optimizer.maximize(init_points=5, n_iter=50, acq='ucb', kappa=5)
    mutating_optimizer.maximize(init_points=5, n_iter=100, acquisition_function=utility)

    lst = mutating_optimizer.space.keys
    lst.append('target')
    df = pd.DataFrame(columns=lst)
    for i, res in enumerate(mutating_optimizer.res):
        _dict = res['params']
        _dict['target'] = res['target']
        row = pd.DataFrame(_dict, index=[0])
        df = pd.concat([df, row], axis=0, ignore_index=True, sort=False)

    df.to_csv(save_file_path, index=False)



