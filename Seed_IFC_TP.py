from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.junctiontree_encoder import JT_SubGraph

from utils.splitter import Splitter
from utils.metrics import Metrics
from utils.Earlystopping import EarlyStopping
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import collate_fraggraphs
from data.dataloading import import_extra_dataset_TP
import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.count_parameters import count_parameters

from networks.AIFC import AIFCNet
from utils.piplines_TP import train_epoch_IFC_TP, evaluate_IFC_TP
from utils.Set_Seed_Reproducibility import set_seed

params = {}
net_params = {}
params['init_lr'] = 10 ** -3
params['min_lr'] = 1e-9
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.8
params['lr_schedule_patience'] = 10
params['earlystopping_patience'] = 50
params['max_epoch'] = 600

net_params['num_atom_type'] = 46
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 128
net_params['num_heads'] = 1
net_params['dropout'] = 0
net_params['depth'] = 3
net_params['layers'] = 3
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cuda'
dataset_list = ['viscosity']


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
    dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer, classic_mol_featurizer, cache_file, load=False
                                 , error_log=error_log_path, fragmentation=fragmentation)

    splitter = Splitter(dataset)
    set_seed(seed=1000)
    rows = []
    file_path = os.path.realpath('./output')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    save_file_path = os.path.join(file_path, '{}_{}_{}'.format(params['Dataset'], 'AIFCNet',
                                                               time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
    result_df = pd.DataFrame(columns=['seed', 'train_R2', 'val_R2', 'test_R2', 'all_R2', 'train_MAE', 'val_MAE', 'test_MAE', 'all_MAE',
                       'train_RMSE', 'val_RMSE', 'test_RMSE', 'all_RMSE'])
    for j in range(0, 50):
        seed = np.random.randint(1, 5000)
        set_seed(seed=1000)

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=seed, frac_train=0.8, frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)
        test_loader = DataLoader(test_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)
        raw_loader = DataLoader(raw_set, collate_fn=collate_fraggraphs, batch_size=1024, shuffle=False)

        model = AIFCNet(net_params).to(device='cuda')
        # model = IFCGATNet(net_params).to(device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                               patience=params['lr_schedule_patience'], verbose=False)
        t0 = time.time()
        per_epoch_time = []
        # early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_seed' + params['Dataset'] + 'IFCGAT' + '.pt')
        early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_seed' + params['Dataset'] + 'AIFC' + '.pt')
        with tqdm(range(params['max_epoch'])) as t:
            n_param = count_parameters(model)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                model, epoch_train_loss, epoch_train_metrics = train_epoch_IFC_TP(model, optimizer, scaling, train_loader, n_param)

                epoch_val_loss, epoch_val_metrics = evaluate_IFC_TP(model, scaling, val_loader, n_param)
                epoch_test_loss, epoch_test_metrics = evaluate_IFC_TP(model, scaling, test_loader, n_param)

                t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                               'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'test_loss': epoch_test_loss,
                               'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2, 'test_R2': epoch_test_metrics.R2})
                per_epoch_time.append(time.time() - start)

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

        row = pd.Series({'seed': seed, 'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                         'test_R2': epoch_test_metrics.R2, 'all_R2': epoch_raw_metrics.R2,
                         'train_MAE': epoch_train_metrics.MAE, 'val_MAE': epoch_val_metrics.MAE,
                         'test_MAE': epoch_test_metrics.MAE, 'all_MAE': epoch_raw_metrics.MAE,
                         'train_RMSE': epoch_train_metrics.RMSE, 'val_RMSE': epoch_val_metrics.RMSE,
                         'test_RMSE': epoch_test_metrics.RMSE, 'all_RMSE': epoch_raw_metrics.RMSE})

        result_df = pd.concat([result_df, row.to_frame().T], ignore_index=True)
    print("After loop:", type(result_df))
    result_df.to_csv(save_file_path)



