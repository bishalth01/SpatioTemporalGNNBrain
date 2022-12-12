import os
import pickle
import random
from collections import deque
from sys import exit
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_geometric.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor, XGBModel
import nibabel as nib

from datasets import ABCDDataset, BrainDataset, HCPDataset, UKBDataset, FlattenCorrsDataset
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy, AnalysisType, merge_y_and_others, EncodingStrategy, create_best_encoder_name, \
    SweepType, DatasetType, get_freer_gpu, free_gpu_info, create_name_for_flattencorrs_dataset, \
    create_name_for_xgbmodel, LRScheduler, Optimiser, EarlyStopping, ModelEmaV2, calculate_indegree_histogram

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = torch.nn.MSELoss(reduction='none')

    def forward(self, y_hat, y):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log()  # where(y_hat > 0, torch.zeros_like(y_hat)).log()
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log()  # where(y > 0, torch.zeros_like(y)).log()
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        # print('A', loss.shape)
        # print(loss.shape, loss)
        # loss = torch.sum(loss, dim=1)
        # if not sum_losses:
        #    loss = loss / seq_length.clamp(min=1)
        return loss.mean()


def train_model(model, train_loader, optimizer, config_params, model_ema=None, label_scaler=None):
    pooling_mechanism = config_params['param_pooling']
    device = config_params['device_run']

    model.train()
    loss_all = 0
    loss_all_link = 0
    loss_all_ent = 0
    if label_scaler is None:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    grads = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
            output_batch, link_loss, ent_loss = model(data)
            output_batch = output_batch.clamp(0, 1)  # For NaNs
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch) # For NaNs
            loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
            loss_b_link = link_loss
            loss_b_ent = ent_loss
        else:
            output_batch = model(data)
            output_batch = output_batch.clamp(0, 1) # For NaNs
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch) # For NaNs
            loss = criterion(output_batch, data.y.unsqueeze(1))

        loss.backward()

        if config_params['final_mlp_layers'] == 1:
            grads.extend(model.final_linear.weight.grad.flatten().cpu().tolist())
        else:
            grads.extend(model.final_linear[-1].weight.grad.flatten().cpu().tolist())

        loss_all += loss.item() * data.num_graphs
        if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
            loss_all_link += loss_b_link.item() * data.num_graphs
            loss_all_ent += loss_b_ent.item() * data.num_graphs

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
    print("GRAD", np.mean(grads), np.max(grads), np.std(grads))
    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset), loss_all_link / len(train_loader.dataset), loss_all_ent / len(
        train_loader.dataset)


def return_regressor_metrics(labels, pred_prob, label_scaler=None, loss_value=None, link_loss_value=None,
                             ent_loss_value=None):
    if label_scaler is not None:
        labels = label_scaler.inverse_transform(labels.reshape(-1, 1))[:, 0]
        pred_prob = label_scaler.inverse_transform(pred_prob.reshape(-1, 1))[:, 0]

    print('First 5 values:', labels.shape, labels[:5], pred_prob.shape, pred_prob[:5])
    r2 = r2_score(labels, pred_prob)
    r = stats.pearsonr(labels, pred_prob)[0]

    return {'loss': loss_value,
            'link_loss': link_loss_value,
            'ent_loss': ent_loss_value,
            'r2': r2,
            'r': r
            }


def return_classifier_metrics(labels, pred_binary, pred_prob, loss_value=None, link_loss_value=None,
                              ent_loss_value=None,
                              flatten_approach: bool = False):
    roc_auc = roc_auc_score(labels, pred_prob)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    report = classification_report(labels, pred_binary, output_dict=True, zero_division=0)

    if not flatten_approach:
        sens = report['1.0']['recall']
        spec = report['0.0']['recall']
    else:
        sens = report['1']['recall']
        spec = report['0']['recall']

    return {'loss': loss_value,
            'link_loss': link_loss_value,
            'ent_loss': ent_loss_value,
            'auc': roc_auc,
            'acc': acc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec
            }


def evaluate_model(model, loader, config_params, label_scaler=None):
    pooling_mechanism = config_params['param_pooling']
    device = config_params['device_run']

    model.eval()
    if label_scaler is None:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    predictions = []
    labels = []
    test_error = 0
    test_link_loss = 0
    test_ent_loss = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
                output_batch, link_loss, ent_loss = model(data)
                output_batch = output_batch.clamp(0, 1)  # For NaNs
                output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
                # output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
                loss_b_link = link_loss
                loss_b_ent = ent_loss
            else:
                output_batch = model(data)
                output_batch = output_batch.clamp(0, 1)  # For NaNs
                output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
                # output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y.unsqueeze(1))

            test_error += loss.item() * data.num_graphs
            if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
                test_link_loss += loss_b_link.item() * data.num_graphs
                test_ent_loss += loss_b_ent.item() * data.num_graphs

            pred = output_batch.flatten().detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    if label_scaler is None:
        pred_binary = np.where(predictions > 0.5, 1, 0)
        return return_classifier_metrics(labels, pred_binary, predictions,
                                         loss_value=test_error / len(loader.dataset),
                                         link_loss_value=test_link_loss / len(loader.dataset),
                                         ent_loss_value=test_ent_loss / len(loader.dataset))
    else:
        return return_regressor_metrics(labels, predictions,
                                        label_scaler=label_scaler,
                                        loss_value=test_error / len(loader.dataset),
                                        link_loss_value=test_link_loss / len(loader.dataset),
                                        ent_loss_value=test_ent_loss / len(loader.dataset))


def training_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader, optimizer,
                  model_ema, config_params, label_scaler=None):
    loss, link_loss, ent_loss = train_model(model, train_loader, optimizer, config_params=config_params,
                                            model_ema=model_ema, label_scaler=label_scaler)
    train_metrics = evaluate_model(model, train_loader, config_params=config_params, label_scaler=label_scaler)
    if model_ema is not None:
        val_metrics = evaluate_model(model_ema.module, val_loader, config_params=config_params, label_scaler=label_scaler)
    else:
        val_metrics = evaluate_model(model, val_loader, config_params=config_params, label_scaler=label_scaler)

    if label_scaler is None:
        print(
            '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
            ' {:.4f} '.format(outer_split_no, inner_split_no, epoch, train_metrics['loss'], val_metrics['loss'],
                              train_metrics['auc'], val_metrics['auc'],
                              train_metrics['acc'], val_metrics['acc'],
                              train_metrics['f1'], val_metrics['f1']))
        log_dict = {
            f'train_loss{inner_split_no}': train_metrics['loss'], f'val_loss{inner_split_no}': val_metrics['loss'],
            f'train_auc{inner_split_no}': train_metrics['auc'], f'val_auc{inner_split_no}': val_metrics['auc'],
            f'train_acc{inner_split_no}': train_metrics['acc'], f'val_acc{inner_split_no}': val_metrics['acc'],
            f'train_sens{inner_split_no}': train_metrics['sensitivity'],
            f'val_sens{inner_split_no}': val_metrics['sensitivity'],
            f'train_spec{inner_split_no}': train_metrics['specificity'],
            f'val_spec{inner_split_no}': val_metrics['specificity'],
            f'train_f1{inner_split_no}': train_metrics['f1'], f'val_f1{inner_split_no}': val_metrics['f1']
        }
    else:
        print(
            '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, R2: {:.4f} / {:.4f}, R: {:.4f} / {:.4f}'
            ''.format(outer_split_no, inner_split_no, epoch, train_metrics['loss'], val_metrics['loss'],
                      train_metrics['r2'], val_metrics['r2'],
                      train_metrics['r'], val_metrics['r']))
        log_dict = {
            f'train_loss{inner_split_no}': train_metrics['loss'], f'val_loss{inner_split_no}': val_metrics['loss'],
            f'train_r2{inner_split_no}': train_metrics['r2'], f'val_r2{inner_split_no}': val_metrics['r2'],
            f'train_r{inner_split_no}': train_metrics['r'], f'val_r{inner_split_no}': val_metrics['r']
        }

    if config_params['param_pooling'] in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
        log_dict[f'train_link_loss{inner_split_no}'] = link_loss
        log_dict[f'val_link_loss{inner_split_no}'] = val_metrics['link_loss']
        log_dict[f'train_ent_loss{inner_split_no}'] = ent_loss
        log_dict[f'val_ent_loss{inner_split_no}'] = val_metrics['ent_loss']

    wandb.log(log_dict)

    return val_metrics


def create_fold_generator(dataset: BrainDataset, config_params: Dict[str, Any], num_splits: int):
    if config_params['dataset_type'] == DatasetType.HCP:

        skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
        merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
                                           torch.cat([data.index for data in dataset], dim=0))
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.hcp_id.item() for data in dataset])
    
    elif config_params['dataset_type'] == DatasetType.ABCD:

        skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
        merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
                                           torch.cat([data.index for data in dataset], dim=0))
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.abcd_id.item() for data in dataset])
    else:

        sexes = []
        bmis = []
        ages = []
        for data in dataset:
            if config_params['analysis_type'] == AnalysisType.FLATTEN_CORRS:
                sexes.append(data.sex.item())
                ages.append(data.age.item())
                bmis.append(data.bmi.item())
            elif config_params['target_var'] == 'gender':
                sexes.append(data.y.item())
                ages.append(data.age.item())
                bmis.append(data.bmi.item())
            elif config_params['target_var'] == 'age':
                sexes.append(data.sex.item())
                ages.append(data.y.item())
                bmis.append(data.bmi.item())
            elif config_params['target_var'] == 'bmi':
                sexes.append(data.sex.item())
                ages.append(data.age.item())
                bmis.append(data.y.item())
        bmis = pd.qcut(bmis, 7, labels=False)
        bmis[np.isnan(bmis)] = 7
        ages = pd.qcut(ages, 7, labels=False)
        strat_labels = LabelEncoder().fit_transform([f'{sexes[i]}{ages[i]}{bmis[i]}' for i in range(len(dataset))])

        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1111)
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  strat_labels)

    return skf_generator


def generate_dataset(config_params: Dict[str, Any]) -> Union[BrainDataset, FlattenCorrsDataset]:
    if config_params['analysis_type'] == AnalysisType.FLATTEN_CORRS:
        name_dataset = create_name_for_flattencorrs_dataset(config_params)
        print("Going for", name_dataset)
        dataset = FlattenCorrsDataset(root=name_dataset,
                                      num_nodes=config_params['num_nodes'],
                                      connectivity_type=config_params['param_conn_type'],
                                      analysis_type=config_params['analysis_type'],
                                      dataset_type=config_params['dataset_type'],
                                      time_length=config_params['time_length'])
    else:
        name_dataset = create_name_for_brain_dataset(num_nodes=config_params['num_nodes'],
                                                     time_length=config_params['time_length'],
                                                     target_var=config_params['target_var'],
                                                     threshold=config_params['param_threshold'],
                                                     normalisation=config_params['param_normalisation'],
                                                     connectivity_type=config_params['param_conn_type'],
                                                     analysis_type=config_params['analysis_type'],
                                                     encoding_strategy=config_params['param_encoding_strategy'],
                                                     dataset_type=config_params['dataset_type'],
                                                     edge_weights=config_params['edge_weights'])
        print("Going for", name_dataset)
        class_dataset = HCPDataset if config_params['dataset_type'] == DatasetType.HCP else ABCDDataset if config_params['dataset_type']== DatasetType.ABCD else UKBDataset
        dataset = class_dataset(root=name_dataset,
                                target_var=config_params['target_var'],
                                num_nodes=config_params['num_nodes'],
                                threshold=config_params['param_threshold'],
                                connectivity_type=config_params['param_conn_type'],
                                normalisation=config_params['param_normalisation'],
                                analysis_type=config_params['analysis_type'],
                                encoding_strategy=config_params['param_encoding_strategy'],
                                time_length=config_params['time_length'],
                                edge_weights=config_params['edge_weights'])

    return dataset


def generate_xgb_model(config_params: Dict[str, Any]) -> XGBModel:
    if config_params['target_var'] == 'gender':
        model = XGBClassifier(subsample=config_params['subsample'],
                              learning_rate=config_params['learning_rate'],
                              max_depth=config_params['max_depth'],
                              min_child_weight=config_params['min_child_weight'],
                              colsample_bytree=config_params['colsample_bytree'],
                              colsample_bynode=config_params['colsample_bynode'],
                              colsample_bylevel=config_params['colsample_bylevel'],
                              n_estimators=config_params['n_estimators'],
                              gamma=config_params['gamma'],
                              n_jobs=-1,
                              random_state=1111)
    else:
        model = XGBRegressor(subsample=config_params['subsample'],
                             learning_rate=config_params['learning_rate'],
                             max_depth=config_params['max_depth'],
                             min_child_weight=config_params['min_child_weight'],
                             colsample_bytree=config_params['colsample_bytree'],
                             colsample_bynode=config_params['colsample_bynode'],
                             colsample_bylevel=config_params['colsample_bylevel'],
                             n_estimators=config_params['n_estimators'],
                             gamma=config_params['gamma'],
                             n_jobs=-1,
                             random_state=1111)
    return model


def generate_st_model(config_params: Dict[str, Any], for_test: bool = False) -> SpatioTemporalModel:
    if config_params['param_encoding_strategy'] in [EncodingStrategy.NONE, EncodingStrategy.STATS]:
        encoding_model = None
    else:
        if config_params['param_encoding_strategy'] == EncodingStrategy.AE3layers:
            pass  # from encoders import AE  # Necessary to torch.load
        elif config_params['param_encoding_strategy'] == EncodingStrategy.VAE3layers:
            pass  # from encoders import VAE  # Necessary to torch.load
        encoding_model = torch.load(create_best_encoder_name(ts_length=config_params['time_length'],
                                                             outer_split_num=outer_split_num,
                                                             encoder_name=config_params['param_encoding_strategy'].value))

    model = SpatioTemporalModel(config_params=config_params,
                                encoding_model=encoding_model
                                ).to(config_params['device_run'])

    if not for_test:
        #wandb.watch(model, log='all')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable params:", trainable_params)
    # elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
    #    model = XGBClassifier(n_jobs=-1, seed=1111, random_state=1111, **params)
    return model


def fit_xgb_model(out_fold_num: int, in_fold_num: int, config_params: Dict[str, Any], model: XGBModel,
                  X_train_in: FlattenCorrsDataset, X_val_in: FlattenCorrsDataset) -> Dict:
    model_saving_path = create_name_for_xgbmodel(model=model,
                                                 outer_split_num=out_fold_num,
                                                 inner_split_num=in_fold_num,
                                                 config_params=config_params
                                                 )

    train_arr = np.array([data.x.numpy() for data in X_train_in])
    val_arr = np.array([data.x.numpy() for data in X_val_in])

    if config_params['target_var'] == 'gender':
        y_train = [int(data.sex.item()) for data in X_train_in]
        y_val = [int(data.sex.item()) for data in X_val_in]
    elif config_params['target_var'] == 'age':
        # np.array() because of printing calls in the regressor_metrics function
        y_train = np.array([float(data.age.item()) for data in X_train_in])
        y_val = np.array([float(data.age.item()) for data in X_val_in])

    model.fit(train_arr, y_train, callbacks=[wandb.xgboost.wandb_callback()])

    pickle.dump(model, open(model_saving_path, "wb"))

    if config_params['target_var'] == 'gender':
        train_metrics = return_classifier_metrics(y_train,
                                                  pred_prob=model.predict_proba(train_arr)[:, 1],
                                                  pred_binary=model.predict(train_arr),
                                                  flatten_approach=True)
        val_metrics = return_classifier_metrics(y_val,
                                                pred_prob=model.predict_proba(val_arr)[:, 1],
                                                pred_binary=model.predict(val_arr),
                                                flatten_approach=True)

        print('{:1d}-{:1d}: Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
              ' {:.4f} '.format(out_fold_num, in_fold_num,
                                train_metrics['auc'], val_metrics['auc'],
                                train_metrics['acc'], val_metrics['acc'],
                                train_metrics['f1'], val_metrics['f1']))
        wandb.log({
            f'train_auc{in_fold_num}': train_metrics['auc'], f'val_auc{in_fold_num}': val_metrics['auc'],
            f'train_acc{in_fold_num}': train_metrics['acc'], f'val_acc{in_fold_num}': val_metrics['acc'],
            f'train_sens{in_fold_num}': train_metrics['sensitivity'],
            f'val_sens{in_fold_num}': val_metrics['sensitivity'],
            f'train_spec{in_fold_num}': train_metrics['specificity'],
            f'val_spec{in_fold_num}': val_metrics['specificity'],
            f'train_f1{in_fold_num}': train_metrics['f1'], f'val_f1{in_fold_num}': val_metrics['f1']
        })
    else:
        train_metrics = return_regressor_metrics(y_train,
                                                 pred_prob=model.predict(train_arr))
        val_metrics = return_regressor_metrics(y_val,
                                               pred_prob=model.predict(val_arr))

        print('{:1d}-{:1d}: R2: {:.4f} / {:.4f}, R: {:.4f} / {:.4f}'.format(out_fold_num, in_fold_num,
                                                                            train_metrics['r2'], val_metrics['r2'],
                                                                            train_metrics['r'], val_metrics['r']))
        wandb.log({
            f'train_r2{in_fold_num}': train_metrics['r2'], f'val_r2{in_fold_num}': val_metrics['r2'],
            f'train_r{in_fold_num}': train_metrics['r'], f'val_r{in_fold_num}': val_metrics['r']
        })

    return val_metrics


def fit_st_model(out_fold_num: int, in_fold_num: int, config_params: Dict[str, Any], model: SpatioTemporalModel,
                 X_train_in: BrainDataset, X_val_in: BrainDataset, label_scaler: MinMaxScaler = None) -> Dict:
    train_in_loader = DataLoader(X_train_in, batch_size=config_params['batch_size'], shuffle=True)#, **kwargs_dataloader)
    val_loader = DataLoader(X_val_in, batch_size=config_params['batch_size'], shuffle=False)#, **kwargs_dataloader)

    ###########
    ## OPTIMISER
    ###########
    if config_params['optimiser'] == Optimiser.SGD:
        print('--> OPTIMISER: SGD')
        optimizer = torch.optim.SGD(model.parameters(),
                                     lr=config_params['param_lr'],
                                     weight_decay=config_params['param_weight_decay'],
                                    )
    elif config_params['optimiser'] == Optimiser.ADAM:
        print('--> OPTIMISER: Adam')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config_params['param_lr'],
                                     weight_decay=config_params['param_weight_decay'])
    elif config_params['optimiser'] == Optimiser.ADAMW:
        print('--> OPTIMISER: AdamW')
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=config_params['param_lr'],
                                     weight_decay=config_params['param_weight_decay'])
    elif config_params['optimiser'] == Optimiser.RMSPROP:
        print('--> OPTIMISER: RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config_params['param_lr'],
                                        weight_decay=config_params['param_weight_decay'])

    ###########
    ## LR SCHEDULER
    ###########
    if config_params['lr_scheduler'] == LRScheduler.STEP:
        print('--> LR SCHEDULER: Step')
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config_params['num_epochs'] / 5), gamma=0.1, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, verbose=True)
    elif config_params['lr_scheduler'] == LRScheduler.PLATEAU:
        print('--> LR SCHEDULER: Plateau')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=config_params['early_stop_steps']-2,
                                                               verbose=True)
    elif config_params['lr_scheduler'] == LRScheduler.COS_ANNEALING:
        print('--> LR SCHEDULER: Cosine Annealing')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_params['num_epochs'], verbose=True)
    elif config_params['lr_scheduler'] == LRScheduler.NONE:
        print('--> LR SCHEDULER: None')

    ###########
    ## EARLY STOPPING
    ###########
    model_saving_name = create_name_for_model(config_params=config_params,
                                              model=model,
                                              outer_split_num=out_fold_num,
                                              inner_split_num=in_fold_num,
                                              prefix_location=''
                                              )

    early_stopping = EarlyStopping(patience=config_params['early_stop_steps'], model_saving_name=model_saving_name)

    ###########
    ## Model Exponential Moving Average (EMA)
    ###########
    if config_params['use_ema']:
        print('--> USING EMA (some repetitions on models outputs because of copying)!')
        # Issues with deepcopy() when using weightnorm, therefore, this workaround is needed
        new_model = generate_st_model(config_params, for_test=True)
        new_model.load_state_dict(model.state_dict())
        model_ema = ModelEmaV2(new_model)
    else:
        model_ema = None
    # Only after knowing EMA deep copies "model", I call wandb.watch()
    wandb.watch(model, log='all')

    for epoch in range(config_params['num_epochs'] + 1):
        val_metrics = training_step(out_fold_num,
                                    in_fold_num,
                                    epoch,
                                    model,
                                    train_in_loader,
                                    val_loader,
                                    optimizer,
                                    model_ema=model_ema,
                                    config_params=config_params,
                                    label_scaler=label_scaler)

        # Calling early_stopping() to update best metrics and stoppping state
        if model_ema is not None:
            early_stopping(val_metrics, model_ema.module, label_scaler)
        else:
            early_stopping(val_metrics, model, label_scaler)
        if early_stopping.early_stop:
            print("EARLY STOPPING IT")
            break

        if config_params['lr_scheduler'] in [LRScheduler.STEP, LRScheduler.COS_ANNEALING]:
            scheduler.step()
        elif config_params['lr_scheduler'] == LRScheduler.PLATEAU:
            scheduler.step(val_metrics['loss'])

    # wandb.unwatch()
    return early_stopping.best_model_metrics


def get_empty_metrics_dict(config_params: Dict[str, Any]) -> Dict[str, list]:
    if config_params['target_var'] == 'gender':
        tmp_dict = {'loss': [], 'sensitivity': [], 'specificity': [], 'acc': [], 'f1': [], 'auc': [],
                    'ent_loss': [], 'link_loss': [], 'best_epoch': []}
    else:
        tmp_dict = {'loss': [], 'r2': [], 'r': [], 'ent_loss': [], 'link_loss': []}
    return tmp_dict


def send_inner_loop_metrics_to_wandb(overall_metrics: Dict[str, list]):
    for key, values in overall_metrics.items():
        if len(values) == 0 or values[0] is None:
            continue
        elif len(values) == 1:
            wandb.run.summary[f"mean_val_{key}"] = values[0]
        else:
            wandb.run.summary[f"mean_val_{key}"] = np.mean(values)
            wandb.run.summary[f"std_val_{key}"] = np.std(values)
            wandb.run.summary[f"values_val_{key}"] = values


def update_overall_metrics(overall_metrics: Dict[str, list], inner_fold_metrics: Dict[str, float]):
    for key, value in inner_fold_metrics.items():
        overall_metrics[key].append(value)


def send_global_results(test_metrics: Dict[str, float]):
    for key, value in test_metrics.items():
        wandb.run.summary[f"values_test_{key}"] = value


if __name__ == '__main__':
    # Because of strange bug with symbolic links in server
    os.environ['WANDB_DISABLE_CODE'] = 'false'
    #file_check = np.load('/data/users2/umahmood1/ICABrainGNN/BrainGNN/DataandLabels/HCP_AllData.npz')
    #file_check = nib.load(f'/data/qneuromark/Results/ICA/HCP/REST1_LR/cleaned_HCP1_sub035_timecourses_ica_s1_.nii')

    #wandb.init(project='hcp_test', save_code=True, config="wandb_sweeps/hcpconfigcustom.yaml")

    wandb.init(project='hcp_test', save_code=True)
    #wandb.init(project='hcp_custom', save_code=True)
    #wandb.init(project='st_extra', save_code=True)
    config = wandb.config

    print('Config file from wandb:', config)

    torch.manual_seed(1)
    np.random.seed(1111)
    random.seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Making a single variable for each argument
    config_params: Dict[str, Any] = {
        'analysis_type': AnalysisType(config.analysis_type),
        'dataset_type': DatasetType(config.dataset_type),
        'num_nodes': config.num_nodes,
        'param_conn_type': ConnType(config.conn_type),
        'split_to_test': config.fold_num,
        'target_var': config.target_var,
        'time_length': config.time_length,
    }
    if config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
        config_params['batch_size'] = config.batch_size
        config_params['device_run'] = f'cuda:{get_freer_gpu()}'
        config_params['early_stop_steps'] = config.early_stop_steps
        config_params['edge_weights'] = config.edge_weights
        config_params['model_with_sigmoid'] = True
        config_params['num_epochs'] = config.num_epochs
        config_params['param_activation'] = config.activation
        config_params['param_channels_conv'] = config.channels_conv
        config_params['param_conv_strategy'] = ConvStrategy(config.conv_strategy)
        config_params['param_dropout'] = config.dropout
        config_params['param_encoding_strategy'] = EncodingStrategy(config.encoding_strategy)
        config_params['param_lr'] = config.lr
        config_params['param_normalisation'] = Normalisation(config.normalisation)
        config_params['param_num_gnn_layers'] = config.num_gnn_layers
        config_params['param_pooling'] = PoolingStrategy(config.pooling)
        config_params['param_threshold'] = config.threshold
        config_params['param_weight_decay'] = config.weight_decay
        config_params['sweep_type'] = SweepType(config.sweep_type)
        config_params['temporal_embed_size'] = config.temporal_embed_size

        config_params['ts_spit_num'] = int(4800 / config_params['time_length'])

        # Not sure whether this makes a difference with the cuda random issues, but it was in the examples :(
        #kwargs_dataloader = {'num_workers': 1, 'pin_memory': True} if config_params['device_run'].startswith('cuda') else {}

        # Definitions depending on sweep_type
        config_params['param_gat_heads'] = 0
        if config_params['sweep_type'] == SweepType.GAT:
            config_params['param_gat_heads'] = config.gat_heads


        # TCN components
        config_params['tcn_depth'] = config.tcn_depth
        config_params['tcn_kernel'] = config.tcn_kernel
        config_params['tcn_hidden_units'] = config.tcn_hidden_units
        config_params['tcn_final_transform_layers'] = config.tcn_final_transform_layers
        config_params['tcn_norm_strategy'] = config.tcn_norm_strategy

        # Training characteristics
        config_params['lr_scheduler'] = LRScheduler(config.lr_scheduler)
        config_params['optimiser'] = Optimiser(config.optimiser)
        config_params['use_ema'] = config.use_ema

        # Node model and final hyperparameters
        config_params['nodemodel_aggr'] = config.nodemodel_aggr
        config_params['nodemodel_scalers'] = config.nodemodel_scalers
        config_params['nodemodel_layers'] = config.nodemodel_layers
        config_params['final_mlp_layers'] = config.final_mlp_layers

        # DiffPool specific stuff
        if config_params['param_pooling'] in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
            config_params['dp_perc_retaining'] = config.dp_perc_retaining
            config_params['dp_norm'] = config.dp_norm

    elif config_params['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
        config_params['device_run'] = 'cpu'
        config_params['colsample_bylevel'] = config.colsample_bylevel
        config_params['colsample_bynode'] = config.colsample_bynode
        config_params['colsample_bytree'] = config.colsample_bytree
        config_params['gamma'] = config.gamma
        config_params['learning_rate'] = config.learning_rate
        config_params['max_depth'] = config.max_depth
        config_params['min_child_weight'] = config.min_child_weight
        config_params['n_estimators'] = config.n_estimators
        config_params['subsample'] = config.subsample

    N_OUT_SPLITS: int = 5
    N_INNER_SPLITS: int = 5

    # Handling inputs and what is possible
    if config_params['analysis_type'] not in [AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL,
                                        AnalysisType.ST_MULTIMODAL_AVG, AnalysisType.ST_UNIMODAL_AVG,
                                        AnalysisType.FLATTEN_CORRS]:
        print('Not yet ready for this analysis type at the moment')
        exit(-1)

    print('This run will not be deterministic')
    if config_params['target_var'] not in ['gender', 'age', 'bmi']:
        print('Unrecognised target_var')
        exit(-1)

    config_params['multimodal_size'] = 0
    #if config_params['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    #    config_params['multimodal_size'] = 10
    #elif config_params['analysis_type'] == AnalysisType.ST_UNIMODAL:
    #    config_params['multimodal_size'] = 0

    if config_params['target_var'] in ['age', 'bmi']:
        config_params['model_with_sigmoid'] = False

    print('Resulting config_params:', config_params)
    # DATASET
    dataset = generate_dataset(config_params)

    skf_outer_generator = create_fold_generator(dataset, config_params, N_OUT_SPLITS)

    # Getting train / test folds
    outer_split_num: int = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1
        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != config_params['split_to_test']:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        break

    scaler_labels = None
    # Scaling for regression problem
    if config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL] and \
            config_params['target_var'] in ['age', 'bmi']:
        print('Mean of distribution BEFORE scaling:', np.mean([data.y.item() for data in X_train_out]),
              '/', np.mean([data.y.item() for data in X_test_out]))
        scaler_labels = MinMaxScaler().fit(np.array([data.y.item() for data in X_train_out]).reshape(-1, 1))
        for elem in X_train_out:
            elem.y[0] = scaler_labels.transform([elem.y.numpy()])[0, 0]
        for elem in X_test_out:
            elem.y[0] = scaler_labels.transform([elem.y.numpy()])[0, 0]

    # Train / test sets defined, running the rest
    print('Size is:', len(X_train_out), '/', len(X_test_out))
    if config_params['analysis_type'] == AnalysisType.FLATTEN_CORRS:
        print('Positive sex classes:', sum([data.sex.item() for data in X_train_out]),
              '/', sum([data.sex.item() for data in X_test_out]))
        print('Mean age distribution:', np.mean([data.age.item() for data in X_train_out]),
              '/', np.mean([data.age.item() for data in X_test_out]))
    elif config_params['target_var'] in ['age', 'bmi']:
        print('Mean of distribution', np.mean([data.y.item() for data in X_train_out]),
              '/', np.mean([data.y.item() for data in X_test_out]))
    else:  # target_var == gender
        print('Positive classes:', sum([data.y.item() for data in X_train_out]),
              '/', sum([data.y.item() for data in X_test_out]))

    skf_inner_generator = create_fold_generator(X_train_out, config_params, N_INNER_SPLITS)

    #################
    # Main inner-loop
    #################
    overall_metrics: Dict[str, list] = get_empty_metrics_dict(config_params)
    inner_loop_run: int = 0
    for inner_train_index, inner_val_index in skf_inner_generator:
        inner_loop_run += 1

        X_train_in = X_train_out[torch.tensor(inner_train_index)]
        X_val_in = X_train_out[torch.tensor(inner_val_index)]
        print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
        if config_params['analysis_type'] == AnalysisType.FLATTEN_CORRS:
            print("Inner Positive sex classes:", sum([data.sex.item() for data in X_train_in]),
                  "/", sum([data.sex.item() for data in X_val_in]))
            print('Mean age distribution:', np.mean([data.age.item() for data in X_train_in]),
                  '/', np.mean([data.age.item() for data in X_val_in]))
        elif config_params['target_var'] in ['age', 'bmi']:
            print('Mean of distribution', np.mean([data.y.item() for data in X_train_in]),
                  '/', np.mean([data.y.item() for data in X_val_in]))
        else:
            print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
                  "/", sum([data.y.item() for data in X_val_in]))

        config_params['dataset_indegree'] = calculate_indegree_histogram(X_train_in)
        print(f'--> Indegree distribution: {config_params["dataset_indegree"]}')

        if config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
            model: SpatioTemporalModel = generate_st_model(config_params)
        elif config_params['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
            model: XGBModel = generate_xgb_model(config_params)
        else:
            model = None

        if config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
            inner_fold_metrics = fit_st_model(out_fold_num=config_params['split_to_test'],
                                              in_fold_num=inner_loop_run,
                                              config_params=config_params,
                                              model=model,
                                              X_train_in=X_train_in,
                                              X_val_in=X_val_in,
                                              label_scaler=scaler_labels)

        elif config_params['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
            inner_fold_metrics = fit_xgb_model(out_fold_num=config_params['split_to_test'],
                                               in_fold_num=inner_loop_run,
                                               config_params=config_params,
                                               model=model,
                                               X_train_in=X_train_in,
                                               X_val_in=X_val_in)
        update_overall_metrics(overall_metrics, inner_fold_metrics)

        # One inner loop only
        #if config_params['dataset_type'] == DatasetType.UKB and config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL,
        #                                                                               AnalysisType.ST_MULTIMODAL]:
        #    break
        # One inner loop no matter what analysis type for more systematic comparison
        break

    send_inner_loop_metrics_to_wandb(overall_metrics)
    print('Overall inner loop results:', overall_metrics)

    #############################################
    # Final metrics on test set, calculated already for being easy to get the metrics on the best model later
    # Getting best model of the run
    inner_fold_for_val: int = 1
    if config_params['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
        model: SpatioTemporalModel = generate_st_model(config_params, for_test=True)

        model_saving_path: str = create_name_for_model(config_params=config_params,
                                                       model=model,
                                                       outer_split_num=config_params['split_to_test'],
                                                       inner_split_num=inner_fold_for_val,
                                                       prefix_location='')

        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, model_saving_path)))
        model.eval()

        # Calculating on test set
        test_out_loader = DataLoader(X_test_out, batch_size=config_params['batch_size'], shuffle=False)#, **kwargs_dataloader)

        test_metrics = evaluate_model(model, test_out_loader, config_params=config_params, label_scaler=scaler_labels)
        print(test_metrics)

        if scaler_labels is None:
            print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                            test_metrics['sensitivity'], test_metrics['specificity']))
        else:
            print('{:1d}-Final: {:.7f}, R2: {:.4f}, R: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['r2'], test_metrics['r']))
    elif config_params['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
        model: XGBModel = generate_xgb_model(config_params)
        model_saving_path = create_name_for_xgbmodel(model=model,
                                                     outer_split_num=config_params['split_to_test'],
                                                     inner_split_num=inner_fold_for_val,
                                                     config_params=config_params
                                                     )
        model = pickle.load(open(model_saving_path, "rb"))
        test_arr = np.array([data.x.numpy() for data in X_test_out])

        if config_params['target_var'] == 'gender':
            y_test = [int(data.sex.item()) for data in X_test_out]
            test_metrics = return_classifier_metrics(y_test,
                                                     pred_prob=model.predict_proba(test_arr)[:, 1],
                                                     pred_binary=model.predict(test_arr),
                                                     flatten_approach=True)
            print(test_metrics)

            print('{:1d}-Final: Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
                  ''.format(outer_split_num, test_metrics['auc'], test_metrics['acc'],
                            test_metrics['sensitivity'], test_metrics['specificity']))
        elif config_params['target_var'] == 'age':
            # np.array() because of printing calls in the regressor_metrics function
            y_test = np.array([float(data.age.item()) for data in X_test_out])
            test_metrics = return_regressor_metrics(y_test,
                                                    pred_prob=model.predict(test_arr))
            print(test_metrics)
            print('{:1d}-Final: R2: {:.4f}, R: {:.4f}'.format(outer_split_num,
                                                              test_metrics['r2'],
                                                              test_metrics['r']))

    send_global_results(test_metrics)

    #if config_params['device_run'] == 'cuda:0':
    #    free_gpu_info()
