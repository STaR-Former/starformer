import matplotlib.pyplot as plt
import logging
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from typing import Any, Mapping, Union, List, Tuple
from sklearn.metrics import fbeta_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import PackedSequence
from lightning.pytorch.utilities import rank_zero_only

import wandb


from ..util import DatasetOptions, ModelOptions
from ..util.dataset import BaseData
from src.nn import MTMLoss, DRMLoss, DRMContrastiveLoss


__all__ = ["CentralizedModel"]

"""
refactor probably move to runtime folder to be honest
"""


class CentralizedModel(L.LightningModule):
    def __init__(self, config, model: nn.Module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.model = model
        self.save_hyperparameters(ignore=['model', 'criterion', 'config'])
        self.configure_cli_logger()
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # log confgis in wandb
            if self.config.logger.name == 'wandb':
                self.log_config_to_wandb()
                
            self.configure_criterion()
            self.logger.log_hyperparams(self.hparams)

        return super().setup(stage)

    @rank_zero_only
    def log_config_to_wandb(self,):
        self.trainer.logger.experiment.config.update(
            OmegaConf.to_container(self.config, resolve=True)
        )

    #def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    #    print('loss backward')
    #    return super().backward(loss, *args, **kwargs)

    #def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        #print('training step')
        return self.__shared_step(batch, batch_idx, mode='train', *args, *kwargs)

    #def training_epoch_end(self, outputs, *args: Any, **kwargs: Any):
    #    torch.cuda.empty_cache()

    #def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        #print('validation step')
        return self.__shared_step(batch, batch_idx, mode='val', *args, *kwargs)

    #def validation_epoch_end(self, outputs, *args: Any, **kwargs: Any):
    #    torch.cuda.empty_cache()

    #def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        #print('test step')
        return self.__shared_step(batch, batch_idx, mode='test', *args, *kwargs)
    
    #def __shared_step(self, batch, batch_idx, mode: str, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
    def __shared_step(self, batch: Tuple[Tensor | PackedSequence] | BaseData,  batch_idx, mode: str, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        if self.config.dataset == DatasetOptions.geolife and \
            self.config.model.name == ModelOptions.starformer:
            return self.__shared_step_ucr_drmt(batch, batch_idx, mode, pred_type='multiclass_cls', *args, **kwargs)

        elif self.config.dataset == DatasetOptions.geolife and \
            self.config.model.name == ModelOptions.starformer:
            return self.__shared_step_ucr_drmt(batch, batch_idx, mode, pred_type='multiclass_cls', *args, **kwargs)
        
        elif self.config.dataset == DatasetOptions.geolife:
            
            if batch.data.data.dtype == torch.float64 and (self.device == torch.device('mps') or self.device == torch.device('mps:0')):
                batch.data = batch.data.float()
                batch.label = batch.label.type(torch.float32)
            
            if self.device == torch.device('mps:0') or self.device == torch.device('mps') \
                    or self.device == torch.device('cuda:0') or self.device == torch.device('cuda'):
                batch.data = batch.data.float().to(self.device)
                batch.label = batch.label.to(self.device)

            
            bs = self.config.training.batch_size
            if self.config.model.name in ModelOptions.lstm:
                outputs = self.model(batch.data)
            else:
                outputs = self.model(batch)
            
            # compute loss
            loss = self.criterion['cross_entropy'](outputs, batch.label.flatten())
    
            # Prediction (get accuracy)
            # acc
            #preds = (outputs > 0.5).int() # converts output probabilities to binary prediction labels, i.e., [0,1]
            outputs_max, outputs_argmax = torch.max(outputs, dim=1)
            acc = torch.sum(torch.eq(outputs_argmax.flatten(), batch.label.flatten().int())) / len(batch.label.flatten())
            # confusion matrix
            text_labels = [0,1,2,3]
            cm_matrix = confusion_matrix(
                batch.label.flatten().detach().cpu().numpy(),
                outputs_argmax.flatten().detach().cpu().numpy(),
                labels=text_labels
                )

            return {
                'loss': loss,
                'preds': outputs_argmax,
                'labels': batch.label,
                'bs': bs,
                'acc': acc,
                'cm': cm_matrix,
                'batch': batch
            }
    
        elif self.config.dataset in DatasetOptions.ucr_uea_multi:
            if self.config.model.name == ModelOptions.starformer:
                return self.__shared_step_ucr_drmt(batch, batch_idx, mode, pred_type='multiclass_cls', *args, **kwargs)
            else:
                return self.__shared_step_ucr(batch, batch_idx, mode, pred_type='multiclass_cls', *args, **kwargs)
        
        elif self.config.dataset in DatasetOptions.ucr_uea_binary:
            if self.config.model.name == ModelOptions.starformer:
                return self.__shared_step_ucr_drmt(batch, batch_idx, mode, pred_type='binary_cls', *args, **kwargs)
            else:
                return self.__shared_step_ucr(batch, batch_idx, mode, pred_type='binary_cls', *args, **kwargs)

        elif self.config.dataset == DatasetOptions.pam and \
            self.config.model.name == ModelOptions.starformer:
            return self.__shared_step_ucr_drmt(batch, batch_idx, mode, pred_type='multiclass_cls', *args, **kwargs)

        raise RuntimeError("Could not identify correct shared step.")
 
    def __shared_step_ucr(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, mode: str, *args: Any, **kwargs: Any):
        pred_type=kwargs['pred_type']
        return_dict = {}

        if batch.data.data.dtype == torch.float64 and (self.device == torch.device('mps') or self.device == torch.device('mps:0')):
                batch.data = batch.data.float()
                batch.label = batch.label.type(torch.float32)
            
        if self.device == torch.device('mps:0') or self.device == torch.device('mps') \
                or self.device == torch.device('cuda:0') or self.device == torch.device('cuda'):
            batch.data = batch.data.float().to(self.device)
            batch.label = batch.label.to(self.device)

        bs = self.config.training.batch_size
        
        return_dict['bs'] = bs
        return_dict['batch'] = batch
        if batch.label.device != self.device:
            batch.label = batch.label.to(self.device)

        #print(batch.label.device, self.device)


        if self.config.model.name in ModelOptions.lstm:
            outputs = self.model(batch.data)
        else:
            outputs = self.model(batch)
        
        # compute loss
        loss = self.__compute_loss(outputs=outputs, batch=batch, pred_type=pred_type)
        #loss = self.criterion['cross_entropy'](outputs, batch.label.flatten())

        return_dict['loss'] = loss

        # Prediction (get accuracy)
        # acc
        preds = self.__compute_predictions(outputs, pred_type=pred_type)
        #outputs_max, outputs_argmax = torch.max(outputs, dim=1)
        acc = torch.sum(torch.eq(preds.flatten(), batch.label.flatten().int())) / len(batch.label.flatten())
        # confusion matrix
        text_labels = list(self.config.datamodule.text_labels)
        cm_matrix = confusion_matrix(
            batch.label.flatten().detach().cpu().numpy(),
            preds.flatten().detach().cpu().numpy(),
            labels=text_labels
            )

        return_dict['preds'] = preds
        return_dict['labels'] = batch.label
        return_dict['acc'] = acc
        return_dict['cm'] = cm_matrix

        return return_dict

    def __shared_step_ucr_drmt(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, mode: str, *args: Any, **kwargs: Any):
        pred_type=kwargs['pred_type']
        return_dict = {}
        #mode = 'train'

        if batch.data.data.dtype == torch.float64 and (self.device == torch.device('mps') or self.device == torch.device('mps:0')):
                batch.data = batch.data.float()
                batch.label = batch.label.type(torch.float32)
            
        if self.device == torch.device('mps:0') or self.device == torch.device('mps') \
                or self.device == torch.device('cuda:0') or self.device == torch.device('cuda'):
            batch.data = batch.data.float().to(self.device)
            batch.label = batch.label.to(self.device)

        bs = self.config.training.batch_size
        
        return_dict['bs'] = bs
        return_dict['batch'] = batch

        if batch.label.device != self.device:
            batch.label = batch.label.to(self.device)

        #print(batch.label.device, self.device)

        if self.config.model.masking is not None and mode != 'test':
            if self.config.loss.loss_fn == 'contrastive_loss':
                out_dict = self.model(x=batch.data, N=batch.seq_len, mode=mode)
                #if True in torch.isnan(out_dict['logits']): # check for nan values and set to 0.0
                #    out_dict['logits'] = torch.where(torch.isnan(out_dict['logits']), torch.tensor(0.0), out_dict['logits'])
                loss_params = {
                    'logits': out_dict['logits'], 
                    'unmasked': out_dict['embedding_cls'], 
                    'masked': out_dict['embedding_masked'], 
                    'y': batch.label, 
                    'pred_type': 'drm'
                }
                if self.config.dataset in [DatasetOptions.geolife]:
                    loss_params['seq_len'] = batch.seq_len
                loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.__compute_loss(
                    **loss_params
                )
                #loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim  = 
                #loss_margin = self.criterion['margin'](unmasked=out_dict['embedding_cls'], masked=out_dict['embedding_masked'], labels=batch.label)

                return_dict['loss'] = loss
                return_dict['loss_ce'] = loss_ce
                return_dict['loss_contrastive'] = loss_contrastive
                return_dict['loss_contrastive_batch_sim'] = loss_contrastive_batch_sim
                return_dict['loss_contrastive_class_sim'] = loss_contrastive_class_sim
                #return_dict['loss_margin'] = loss_margin

            elif self.config.loss.loss_fn == 'drm_loss':
                out_dict = self.model(x=batch.data, N=batch.seq_len, mode=mode)
                #if True in torch.isnan(out_dict['logits']): # check for nan values and set to 0.0
                #    out_dict['logits'] = torch.where(torch.isnan(out_dict['logits']), torch.tensor(0.0), out_dict['logits'])
                loss, loss_ce, loss_mse, loss_mse_masked, loss_mse_unmasked = self.__compute_loss(
                    logits=out_dict['logits'], y=batch.label, x_pred=out_dict['out_representation'], x=batch.data, 
                    padding_mask=out_dict['padding_mask'], sequence_mask=out_dict['sequence_mask'], pred_type='drm')
                
                return_dict['loss'] = loss
                return_dict['loss_ce'] = loss_ce
                return_dict['loss_mse'] = loss_mse
                return_dict['loss_mse_masked'] = loss_mse_masked
                return_dict['loss_mse_unmasked'] = loss_mse_unmasked
            
        else:
            #if True in torch.isnan(out_dict['logits']): # check for nan values and set to 0.0
            #    out_dict['logits'] = torch.where(torch.isnan(out_dict['logits']), torch.tensor(0.0), out_dict['logits'])
            out_dict = self.model(x=batch.data, N=batch.seq_len, mode=mode)
            # compute loss
            #if True in torch.isnan(out_dict['logits']): # check for nan values and set to 0.0
            #    out_dict['logits'] = torch.where(torch.isnan(out_dict['logits']), torch.tensor(0.0), out_dict['logits'])
            loss = self.__compute_loss(logits=out_dict['logits'], batch=batch, pred_type=pred_type)
            
            #loss = self.criterion['cross_entropy'](outputs, batch.label.flatten())
            return_dict['loss'] = loss

        # Prediction (get accuracy)
        # acc
        preds = self.__compute_predictions(out_dict['logits'], pred_type=pred_type)
        #outputs_max, outputs_argmax = torch.max(outputs, dim=1)
        acc = torch.sum(torch.eq(preds.flatten(), batch.label.flatten().int())) / len(batch.label.flatten())
        # confusion matrix
        text_labels = list(self.config.datamodule.text_labels)
        cm_matrix = confusion_matrix(
            batch.label.flatten().detach().cpu().numpy(),
            preds.flatten().detach().cpu().numpy(),
            labels=text_labels
            )

        return_dict['preds'] = preds
        return_dict['labels'] = batch.label
        return_dict['acc'] = acc
        return_dict['cm'] = cm_matrix
        
        #print('forward pass worked')
        return return_dict


    def __compute_loss(self, 
        logits: Tensor, # target scores, no sigmoid applied (no proba's)
        y: Tensor=None, 
        x_pred: Tensor=None, 
        x: Tensor=None, 
        outputs_mtm: Tensor=None, 
        batch=None, 
        padding_mask: Tensor=None, 
        sequence_mask: Tensor=None, 
        unmasked: Tensor=None, 
        masked: Tensor=None,
        pred_type: str='mtm',
        seq_len: Tensor=None):

        if pred_type == 'drm' and self.config.model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'contrastive_loss':
            loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.criterion['contrastive'](
                y_logits=logits, unmasked=unmasked, masked=masked, labels=y, seq_len=seq_len,
            ) 
            return loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim
        
        elif pred_type == 'drm' and self.config.model.name == ModelOptions.starformer and self.config.model.masking is not None:
            loss, loss_ce, loss_mse, loss_mse_masked, loss_mse_unmasked = self.criterion['drm_loss'](
                y_pred=logits, y=y.flatten(), x_pred=x_pred, x=x, padding_mask=padding_mask, sequence_mask=sequence_mask
            )
            return loss, loss_ce, loss_mse, loss_mse_masked, loss_mse_unmasked

        elif pred_type == 'binary_cls':
            if logits.dtype == torch.double:
                labels = batch.label.type(logits.dtype)
            else:
                labels = batch.label
            
            #print(batch.label.dtype, logits.dtype)
            if batch.label.dtype == torch.long or batch.label.dtype == torch.int64:
                labels = batch.label.type(logits.dtype)
            #print(labels.dtype)
            loss = self.criterion['binary_cross_entropy'](logits.flatten(), labels.flatten())
            return loss

        elif pred_type == 'multiclass_cls':
            loss = self.criterion['cross_entropy'](logits, batch.label.flatten())
            return loss

        else:
            raise NotImplementedError(f'{pred_type}')

    def __compute_predictions(self, outputs: Tensor, pred_type: str='binary_cls', ptr: List[int]=None, autoregressive: bool=False):
        # convert logits to probablilities
        if autoregressive:
            last_state_per_sequence = ptr[1:]-1
            auto_regr_logits = outputs[last_state_per_sequence, ...]
            outputs = auto_regr_logits

        if pred_type == 'binary_cls':
            outputs = F.sigmoid(outputs)
            preds = (outputs > 0.5).int() # converts output probabilities to binary prediction labels, i.e., [0,1]
        elif pred_type == 'multiclass_cls':
            outputs = F.softmax(outputs, dim=1)
            outputs_max, outputs_argmax = torch.max(outputs, dim=1)
            preds = outputs_argmax
        else:
            raise NotImplementedError(f'{pred_type}')
        
        return preds
        
    
    def configure_optimizers(self):
        if self.config.optimizer.name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), 
                lr=self.config.training.learning_rate, 
                momentum=self.config.optimizer.momentum
            )
        elif self.config.optimizer.name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), 
                lr=self.config.training.learning_rate
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), 
                lr=self.config.training.learning_rate, 
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.eps,
                weight_decay=self.config.optimizer.weight_decay,
            )
        
        scheduler = None   
        if self.config.callbacks.lr_scheduler.apply:
            if self.config.callbacks.lr_scheduler.name == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode=self.config.callbacks.lr_scheduler.mode,
                    factor=self.config.callbacks.lr_scheduler.factor,
                    patience=self.config.callbacks.lr_scheduler.patience,
                    min_lr=self.config.callbacks.lr_scheduler.min_lr,
                )
            else:
                raise ValueError(f'{self.config.callbacks.lr_scheduler.name} not knonw!')
        if scheduler == None:
            return optimizer
        else:
            return [optimizer], [{
                'scheduler': scheduler,
                'monitor': self.config.callbacks.lr_scheduler.monitor,
            }]
        #{
        #        'optimizer': optimizer,
        #        'scheduler': scheduler,
        #        'monitor': self.config.callbacks.lr_scheduler.monitor,
        #    }
            #[optimizer], [{'scheduler': scheduler, 'monitor': self.config.callbacks.lr_scheduler.monitor}]

    def configure_criterion(self):
        self.criterion = {}
        
        _multi = [DatasetOptions.geolife] 
        _multi.extend(DatasetOptions.ucr_uea_multi)
        _multi.append(DatasetOptions.pam)

        _binary = DatasetOptions.ucr_uea_binary
        
        if isinstance(self.config.loss.loss_fn, str):
            loss_fns = [self.config.loss.loss_fn]

        for loss_fn in loss_fns:
            if loss_fn == 'cross_entropy':
                self.criterion[loss_fn] = nn.CrossEntropyLoss()
            elif loss_fn == 'nll_loss':
                self.criterion[loss_fn] = nn.NLLLoss()
            elif loss_fn == 'binary_cross_entropy':
                self.criterion[loss_fn] = nn.BCEWithLogitsLoss()
            elif loss_fn == 'mean_squarred_error':
                self.criterion[loss_fn] = nn.MSELoss()
            elif loss_fn == 'mtm_loss':
                self.criterion['mtm_loss'] = MTMLoss(weight=self.config.loss.weight, dataset=self.config.loss.dataset)
            elif loss_fn == 'drm_loss':
                loss_kwargs = {
                    'lambda_drm': self.config.loss.lambda_drm, 
                    'lambda_masked': self.config.loss.lambda_masked,
                }

                if self.config.dataset in _multi:
                    loss_kwargs['pred_type'] = 'multiclass_cls'
                elif self.config.dataset in _binary:
                    loss_kwargs['pred_type'] = 'binary_cls'
                
                self.criterion['drm_loss'] = DRMLoss(**loss_kwargs)
                if loss_kwargs['pred_type'] == 'multiclass_cls':
                    self.criterion['cross_entropy'] = nn.CrossEntropyLoss() 
                else:
                    self.criterion['binary_cross_entropy'] = nn.BCEWithLogitsLoss()

            elif loss_fn == 'contrastive_loss':
                loss_kwargs = {
                    'temp': self.config.loss.temp, 
                    'weight_batch_class': self.config.loss.lamdba_sim,
                    'batch_size': self.config.training.batch_size, 
                    'weight_contrastive': self.config.loss.lambda_contrastive 
                }

                if self.config.dataset in _multi:
                    loss_kwargs['pred_type'] = 'multiclass_cls'
                    self.criterion['cross_entropy'] = nn.CrossEntropyLoss() 

                elif self.config.dataset in _binary:
                    loss_kwargs['pred_type'] = 'binary_cls'
                    self.criterion['binary_cross_entropy'] = nn.BCEWithLogitsLoss()

                self.criterion['contrastive'] = DRMContrastiveLoss(**loss_kwargs)
                
            else:
                raise ValueError(f'{self.config.loss.loss_fn} is not available!')
    
    def configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger



def create_masked_trajecotries(batch, N: Tensor, mask_p=0.2):
    # create padding mask
    #padding_mask = (batch.sum(dim=-1) == 0).T
    # get unpadded sequence lengths
    #values, unpadded_seq_lengths = np.unique(torch.where(padding_mask == False)[0].detach().cpu().numpy(), return_counts=True)
    # create indices which will be masked
    #mask_indices = [
    #    torch.randint(0, N, (int(N*mask_p),)).sort().values
    #    for i, N in enumerate(unpadded_seq_lengths)
    #]

    # create indices which will be masked
    mask_indices = [
        torch.randint(0, n.item(), (int(n.item()*mask_p),)).sort().values if int(mask_p) != 1 else torch.arange(0, n.item(),1)
        for i, n in enumerate(N)
    ]
    
    # apply mask
    masked_batch = torch.zeros_like(batch, dtype=batch.dtype, device=batch.device)
    for i, indices in enumerate(mask_indices):
        masked_batch_item = deepcopy(batch[:, i, :]) 
        # create mask
        masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
        masked_batch[:, i, :] = masked_batch_item

    # test 
    for i, indices in enumerate(mask_indices):
        assert torch.all(masked_batch[indices, i, :]==0.0)
    
    return masked_batch
