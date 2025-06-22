import os

import dgl.backend as F
import dgl
import numpy as np
import pandas as pd
import torch

from data.dataloading import import_extra_dataset, import_extra_dataset_TP
from dgl.data.utils import save_graphs, load_graphs
from utils.mol2graph import graph_2_frag, create_channels


class MoleculeCSVDataset(object):
    '''
    :param
    df: pandas.DataFrame
    '''
    def __init__(self, df, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, cache_file_path, load=False, log_every=100, error_log=None, fragmentation=None):
        self.df = df
        self.cache_file_path = cache_file_path
        self.valid_idx = []
        self._prepare(smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, load, log_every, error_log, fragmentation)
        self.whe_frag = False
        if fragmentation is not None:
            self.whe_frag = True
            self._prepare_frag(fragmentation, load, log_every, error_log)
            self._prepare_channel()
        # self.temperature = df['TEMPERATURE']
        # self.pressure = df['PRESSURE']

    def _prepare(self, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, load, log_every, error_log, fragmentation):
        '''
        :param
        '''
        if os.path.exists(self.cache_file_path) and load:
            print('Loading saved dgl graphs ...')
            self.origin_graphs, label_dict = load_graphs(self.cache_file_path)
            self.values = label_dict['values']
            valid_idx = label_dict['valid_idx']
            self.valid_idx = valid_idx.detach().numpy().tolist()
        else:
            print('Preparing dgl by featurizers ...')
            self.origin_graphs = []
            for i, s in enumerate(self.df['SMILES']):
                if (i + 1) % log_every == 0:
                    print('Currently preparing molecule {:d}/{:d}'.format(i + 1, len(self)))
                self.origin_graphs.append(smiles_2_graph(s, atom_featurizer, bond_featurizer, mol_featurizer))

            # Check failed featurization
            # Keep successful featurization
            self.valid_idx = []
            origin_graphs = []
            failed_smiles = []
            for i, g in enumerate(self.origin_graphs):
                if g is not None:
                    self.valid_idx.append(i)
                    origin_graphs.append(g)
                else:
                    failed_smiles.append((i, self.df['SMILES'][i]))

            if error_log is not None:
                if len(failed_smiles) > 0:
                    failed_idx, failed_smis = map(list, zip(*failed_smiles))
                else:
                    failed_idx, failed_smis = [], []
                df = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
                df.to_csv(error_log, index=False)

            self.origin_graphs = origin_graphs
            _label_values = self.df['VALUE']
            self.values = F.zerocopy_from_numpy(np.nan_to_num(_label_values).astype(np.float32))[self.valid_idx]
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(self.cache_file_path, self.origin_graphs, labels={'values': self.values, 'valid_idx': valid_idx})


        self.smiles = [self.df['SMILES'][i] for i in self.valid_idx]
        self.names = [self.df['NAMES'][i] for i in self.valid_idx]
        # _label_temperature = self.df['TEMPERATURE']
        # self.temperature = F.zerocopy_from_numpy(np.nan_to_num(_label_temperature).astype(np.float32))[self.valid_idx]
        # _label_pressure = self.df['PRESSURE']  # TP
        # self.pressure = F.zerocopy_from_numpy(np.nan_to_num(_label_pressure).astype(np.float32))[self.valid_idx]

    def _prepare_frag(self, fragmentation, load, log_every, error_log):
        _frag_cache_file_path = self.cache_file_path + '_frag'
        _motif_cache_file_path = self.cache_file_path + '_motif'
        if os.path.exists(_frag_cache_file_path) and os.path.exists(_motif_cache_file_path) and load:
            print('Loading saved fragments and graphs ...')
            unbatched_frag_graphs, frag_label_dict = load_graphs(_frag_cache_file_path)
            self.motif_graphs, motif_label_dict = load_graphs(_motif_cache_file_path)
            frag_graph_idx = frag_label_dict['frag_graph_idx'].detach().numpy().tolist()
            self.batched_frag_graphs = self.batch_frag_graph(unbatched_frag_graphs, frag_graph_idx)

        else:
            print('Preparing fragmentation ...')
            self.batched_frag_graphs = []
            unbatched_frag_graphs_list = [] # unbatched_* variables prepared for storage of graphs
            self.motif_graphs = []
            self.atom_mask_list = []
            self.frag_flag_list = []
            valid_indices = []
            for i, s in enumerate(self.df['SMILES']):
                if (i + 1) % log_every == 0:
                    print('Currently proceeding fragmentation on molecule {:d}/{:d}'.format(i + 1, len(self)))
                try:
                    frag_graph, motif_graph, atom_mask, frag_flag = graph_2_frag(s, self.origin_graphs[i], fragmentation)
                except:
                    print('Failed to deal with  ', s)
                    continue
                self.batched_frag_graphs.append(dgl.batch(frag_graph))
                unbatched_frag_graphs_list.append(frag_graph)
                self.motif_graphs.append(motif_graph)
                self.atom_mask_list.append(atom_mask)
                self.frag_flag_list.append(frag_flag)
                valid_indices.append(i)

            self.valid_idx = valid_indices
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)

            batched_frag_graphs = []
            unbatched_frag_graphs = []
            motif_graphs = []
            frag_flag_list = []
            frag_failed_smiles = []
            for i, g in enumerate(self.motif_graphs):
                if g is not None:
                    motif_graphs.append(g)
                    frag_flag_list.append(self.frag_flag_list[i])
                    batched_frag_graphs.append(self.batched_frag_graphs[i])
                    unbatched_frag_graphs.append(unbatched_frag_graphs_list[i])
                else:
                    frag_failed_smiles.append((i, self.df['SMILES'][i]))
                    # self.valid_idx.remove(i)

            if len(frag_failed_smiles) > 0:
                failed_idx, failed_smis = map(list, zip(*frag_failed_smiles))
            else:
                failed_idx, failed_smis = [], []
            df_failed = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
            if os.path.exists(error_log):
                df_failed.to_csv(error_log, mode='a', index=False)
            else:
                df_failed.to_csv(error_log, mode='w', index=False)
            self.batched_frag_graphs = batched_frag_graphs
            self.motif_graphs = motif_graphs
            self.frag_flag_list = frag_flag_list
            unbatched_frag_graphs, frag_graph_idx = self.merge_frag_list(unbatched_frag_graphs)
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(_frag_cache_file_path, unbatched_frag_graphs, labels={'values': self.values, 'valid_idx': valid_idx, 'frag_graph_idx': frag_graph_idx})
            save_graphs(_motif_cache_file_path, self.motif_graphs, labels={'values': self.values, 'valid_idx': valid_idx})

    def _prepare_channel(self):
        self.channel_graphs = []
        for _ in range(len(self.df)):
            self.channel_graphs.append(create_channels())

    def __getitem__(self, index):
        if self.whe_frag:
#           return self.origin_graphs[index], self.batched_frag_graphs[index], self.motif_graphs[index], self.channel_graphs[index], self.values[index], self.smiles[index],self.names[index], self.temperature[index], self.pressure[index]
            return self.origin_graphs[index], self.batched_frag_graphs[index], self.motif_graphs[index], \
            self.channel_graphs[index], self.values[index], self.smiles[index],self.names[index]
        else:
            # return self.origin_graphs[index], self.values[index], self.smiles[index], self.names[index], self.temperature[index], self.pressure[index]

            return self.origin_graphs[index], self.values[index], self.smiles[index], self.names[index]
    def __len__(self):
        return len(self.valid_idx)

    def merge_frag_list(self, frag_graphs_list):
        # flatten all fragment lists in self.frag_graphs_lists for saving, [[...], [...], [...], ...] --> [..., ..., ...]
        frag_graphs = []
        idx = []
        for i, item in enumerate(frag_graphs_list):
            for _ in range(len(item)):
                idx.append(i)
            frag_graphs.extend(item)
        idx = torch.Tensor(idx)
        return frag_graphs, idx

    def convert_frag_list(self):
        # convert flattened list into 2-D list [[], [], []], inner list represents small subgraph of fragment while outer list denotes the index of molecule
        frag_graphs_list = [[] for _ in range(len(self))]
        for i, item in enumerate(self.frag_graph_idx):
            frag_graphs_list[int(item)].append(self.frag_graphs[i])
        return frag_graphs_list

    def batch_frag_graph(self, unbatched_graph, frag_graph_idx):
        batched_frag_graphs = []
        for i in range(len(self)):
            batched_frag_graph = dgl.batch([unbatched_graph[idx] for idx, value in enumerate(frag_graph_idx) if int(value) == i])
            batched_frag_graphs.append(batched_frag_graph)
        return batched_frag_graphs


# get frag_flag
if __name__ == '__main__':
    from src.feature.atom_featurizer import classic_atom_featurizer
    from src.feature.bond_featurizer import classic_bond_featurizer
    from src.feature.mol_featurizer import classic_mol_featurizer
    from utils.mol2graph import smiles_2_bigraph
    from utils.junctiontree_encoder import JT_SubGraph
    from plottools.visualize_attention import visualize_frag_weights, print_frag_attentions
    import time
    import csv
    dataset_list = ['tdecomp']
    params = {}
    net_params = {}
    for i in range(len(dataset_list)):
        params['Dataset'] = dataset_list[i]
        # df, scaling, T, P = import_extra_dataset_TP(params)
        df, scaling = import_extra_dataset(params)
    # print(df.columns)
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
    csv_file_path = '../output/frag_flag.csv'
    with open (csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for sublist in dataset.frag_flag_list:
            writer.writerow(sublist)

    attention_frag = []
    file_path = '../datasets/_AIFC_attention_prediction.csv'
    target_path = ''
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            cleaned_row = []
            for element in row[:53]:
                if element.strip():
                    cleaned_row.append(float(element))
            attention_frag.append(cleaned_row)
    # print(attention_frag)
    df_attention = pd.read_csv(file_path, sep=',', encoding='utf-8')
    df_target = pd.read_csv(target_path, sep=',', encoding='utf-8')
    path = 'frag_attetion_visual_AIFC'
    smiles_list = [smiles for smiles in df['SMILES']]

    target_list = [target for target in df_target['VALUE']]
    frag_list = dataset.frag_flag_list
    # print(frag_list)
    predictions_list = [prediction for prediction in df_attention['PREDICTION']]
    attention_list = attention_frag
    atom_mask_list = dataset.atom_mask_list
    visualize_frag_weights(path, smiles_list, target_list, frag_list, predictions_list, attention_list, atom_mask_list)
    print_frag_attentions(path, smiles_list, attention_list, atom_mask_list, frag_list)
