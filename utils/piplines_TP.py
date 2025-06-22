import time
import torch
from metrics import Metrics


def train_epoch_IFC_TP(model, optimizer, scaling, data_loader, n_param=None):
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0

    for batch_data in data_loader:
        batch_origin_graph, batch_frag_graph, batch_motif_graph, targets, smiles, names, temperature, pressure = batch_data
        batch_origin_node = batch_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batch_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_graph = batch_origin_graph.to(device='cuda')

        batch_frag_node = batch_frag_graph.ndata['feat'].to(device='cuda')
        batch_frag_edge = batch_frag_graph.edata['feat'].to(device='cuda')
        batch_frag_graph = batch_frag_graph.to(device='cuda')

        batch_motif_node = batch_motif_graph.ndata['feat'].to(device='cuda')
        batch_motif_edge = batch_motif_graph.edata['feat'].to(device='cuda')
        batch_motif_graph = batch_motif_graph.to(device='cuda')

        temperature_tensor = temperature .to(device='cuda')
        pressure_tensor = pressure.to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, temperature_tensor, pressure_tensor)
        target = targets.float().to(device='cuda')
        loss = model.loss(score, target)
        loss.backward()
        optimizer.step()

        score_list.append(score.detach())
        target_list.append(target.detach())
        epoch_loss += loss.detach().item()

    score_list = torch.cat(score_list, dim=0).cpu()
    target_list = torch.cat(target_list, dim=0).cpu()
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list), scaling.ReScaler(score_list), n_param)
    return model, epoch_loss, epoch_train_metrics


def evaluate_IFC_TP(model, scaling, data_loader, n_param=None, flag=False):

    model.eval()
    score_list = []
    target_list = []
    smiles_list = []
    epoch_loss = 0
    with torch.no_grad():
        for batch_data in data_loader:
            batch_origin_graph, batch_frag_graph, batch_motif_graph, targets, smiles, names, temperature, pressure = batch_data

            batch_origin_node = batch_origin_graph.ndata['feat'].to(device='cuda')
            batch_origin_edge = batch_origin_graph.edata['feat'].to(device='cuda')
            batch_origin_graph = batch_origin_graph.to(device='cuda')

            batch_frag_node = batch_frag_graph.ndata['feat'].to(device='cuda')
            batch_frag_edge = batch_frag_graph.edata['feat'].to(device='cuda')
            batch_frag_graph = batch_frag_graph.to(device='cuda')

            batch_motif_node = batch_motif_graph.ndata['feat'].to(device='cuda')
            batch_motif_edge = batch_motif_graph.edata['feat'].to(device='cuda')
            batch_motif_graph = batch_motif_graph.to(device='cuda')

            temperature_tensor = temperature.to(device='cuda')
            pressure_tensor = pressure.to(device='cuda')


            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, temperature_tensor, pressure_tensor)
            target = targets.float().to(device='cuda')
            loss = model.loss(score, target)

            score_list.append(score.detach())
            target_list.append(target.detach())
            smiles_list.extend(smiles)

            epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0).cpu()
    target_list = torch.cat(target_list, dim=0).cpu()
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list), scaling.ReScaler(score_list), n_param)

    predict = scaling.ReScaler(score_list)
    true = scaling.ReScaler(target_list)
    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_IFC_descriptors_TP(model, scaling, data_loader, n_param=None):
    model.eval()
    score_list = []
    target_list = []
    for batch_data in data_loader:
        batch_origin_graph, batch_frag_graph, batch_motif_graph, targets, smiles, names,temperature, pressure = batch_data
        batch_origin_node = batch_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batch_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_graph = batch_origin_graph.to(device='cuda')

        batch_frag_node = batch_frag_graph.ndata['feat'].to(device='cuda')
        batch_frag_edge = batch_frag_graph.edata['feat'].to(device='cuda')
        batch_frag_graph = batch_frag_graph.to(device='cuda')

        batch_motif_node = batch_motif_graph.ndata['feat'].to(device='cuda')
        batch_motif_edge = batch_motif_graph.edata['feat'].to(device='cuda')
        batch_motif_graph = batch_motif_graph.to(device='cuda')

        temperature_tensor = temperature.to(device='cuda')
        pressure_tensor = pressure.to(device='cuda')

        if True:
            _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, temperature_tensor,
                                           pressure_tensor, get_descriptors=True)

    return smiles, descriptors


def evaluate_IFC_attention_TP(model, scaling, data_loader):
    model.eval()
    score_list = []
    attentions_list = []
    for batch_data in data_loader:
        batch_origin_graph, batch_frag_graph, batch_motif_graph, targets, smiles, names, temperature, pressure = batch_data
        batch_origin_node = batch_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batch_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_global = batch_origin_graph.ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batch_origin_graph.to(device='cuda')

        batch_frag_node = batch_frag_graph.ndata['feat'].to(device='cuda')
        batch_frag_edge = batch_frag_graph.edata['feat'].to(device='cuda')
        batch_frag_graph = batch_frag_graph.to(device='cuda')

        batch_motif_node = batch_motif_graph.ndata['feat'].to(device='cuda')
        batch_motif_edge = batch_motif_graph.edata['feat'].to(device='cuda')
        batch_motif_graph = batch_motif_graph.to(device='cuda')

        temperature_tensor = temperature.to(device='cuda')
        pressure_tensor = pressure.to(device='cuda')

        score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge,
                                         temperature_tensor, pressure_tensor, get_attention=True)
        score_list.append(score.detach())
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list



