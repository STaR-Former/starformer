import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch import Tensor

from typing import Literal


__all__ = [
    "PerSeqMSELoss",
    "MTMLoss",
    'DRMLoss',
    "ContrastiveLoss",
    "MBSLoss",
    "DRMContrastiveLoss"
]

"""
refactor

- create two main loss functions

1. DAReM reconstruction loss 
2. DAReM contrastive loss

"""


class PerSeqMSELoss(nn.Module):
    def __init__(self, debug: bool=False, max_loss: float=10.0, verbose: bool=False):
        super(PerSeqMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.debug = debug
        self.max_loss = max_loss #  for outliners
        self.verbose = verbose
        if debug:
            self.bad_traj = []

    def forward(self, y_pred, batch, verbose: bool=False):
        # x --> [seq_len, bs, D] 
        mse = [
            self.mse_loss(batch.data[:batch.seq_len[i], i, ...], y_pred[:batch.seq_len[i], i, ...]).reshape(1,-1)
            for i in range(batch.batch_size)
        ]
        #mse_y = [
        #    self.mse_loss(batch.data[:batch.seq_len[i], i, :1], y_pred[:batch.seq_len[i], i, :1]).reshape(1,-1)
        #    for i in range(batch.batch_size)
        #]
        if self.debug:
            for idx, loss in enumerate(mse):
                if loss >= self.max_loss:
                    if verbose: print(batch.traj_id[idx], loss, batch.label[idx])
                    self.bad_traj.append((batch.traj_id[idx], loss, batch.label[idx]))
            return torch.concat(mse).mean().squeeze()

        return torch.concat(mse).mean().squeeze()


class MTMLoss(nn.Module):
    def __init__(self, weight: float=1.0, pred_type: Literal['binary_cls', 'multiclass_cls']='binary_cls'):
        super(MTMLoss, self).__init__()
        self.weight = weight
        self.mse = PerSeqMSELoss(debug=False)
        self._pred_type = pred_type

        if self._pred_type == 'multiclass_cls':
            self.ce = nn.CrossEntropyLoss()
        elif self._pred_type == 'multiclass_cls':
            self.ce = nn.BCELoss()
        else:
            raise RuntimeError(f'{self._pred_type} is not implemented!')
    
    def forward(self, y_preds: Tensor, y: Tensor, y_preds_seq: Tensor, batch):
        loss_ce = self.ce(y_preds, y.flatten())
        loss_mse = self.mse(y_preds_seq, batch)
        loss = loss_ce + self.weight*loss_mse
        return loss, loss_ce, loss_mse


class DRMReconstructionLoss(nn.Module):
    def __init__(self, lambda_masked: float=0.5,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mse_masked = nn.MSELoss()
        self.mse_unmasked = nn.MSELoss()
        self._lambda_masked = lambda_masked
        assert 0 < lambda_masked < 1.0
    
    def forward(self, x_pred: Tensor, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor):
        masked_elements = sequence_mask & ~padding_mask
        unmasked_elements = ~sequence_mask & ~padding_mask
        masked_loss = self.mse_masked(x[masked_elements], x_pred[masked_elements])
        unmasked_loss = self.mse_unmasked(x[unmasked_elements], x_pred[unmasked_elements])
        return self._lambda_masked * masked_loss + (1-self._lambda_masked) * unmasked_loss, masked_loss, unmasked_loss


class DRMLoss(nn.Module):
    def __init__(self, lambda_drm: float=1.0, lambda_masked: float=0.5, pred_type: Literal['binary_cls', 'multiclass_cls']='binary_cls', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lambda_drm = lambda_drm
        self.mse = DRMReconstructionLoss(lambda_masked=lambda_masked)
        self._pred_type = pred_type

        if self._pred_type == 'multiclass_cls':
            self.ce = nn.CrossEntropyLoss()
        elif self._pred_type == 'binary_cls':
            self.ce = nn.BCEWithLogitsLoss()
        else:
            raise RuntimeError(f'{self._pred_type} is not implemented!')
    
    def forward(self, y: Tensor, y_pred: Tensor, x: Tensor, x_pred: Tensor, padding_mask: Tensor, sequence_mask: Tensor):
        if self._pred_type == 'binary_cls':
            y_pred = y_pred.flatten()
            if x_pred.dtype == torch.double:
                y = y.type(x_pred.dtype)

        loss_ce = self.ce(y_pred, y.flatten())
        loss_mse, loss_mse_masked, loss_mse_unmasked = self.mse(x_pred, x, padding_mask=padding_mask, sequence_mask=sequence_mask)
        loss = loss_ce + self._lambda_drm*loss_mse
        
        return loss, loss_ce, loss_mse, loss_mse_masked, loss_mse_unmasked


class ContrastiveLoss(nn.Module):
    def __init__(self, temp: float = 0.5, weight: float=0.5, batch_size: int=None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.temp = temp
        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_class_sim_prev = 0.0
        self.batch_size = batch_size
    
    def forward(self, unmasked: Tensor, masked: Tensor, labels: Tensor, seq_len: Tensor=None):
        """
        Args:
            unmasked: unmasked embedding from encoder, [N, B, D]
            masked: masked embedding from encoder, [N, B, D]
        """
        # create a mask if sequences in batch are padded
        if seq_len is not None:
            N, B, D = unmasked.size()
            #bs_range = list(range(b))
            padding_mask = torch.zeros((N, B, 1), dtype=torch.bool, device=unmasked.device)
            for i, seq_len in enumerate(seq_len):
                padding_mask[:seq_len, i, :] = True

            # contrative loss
            # reduce dimensionality, take sum across the sequence
            unmasked_rd = (unmasked*padding_mask).mean(dim=0) # [BS, D]
            masked_rd = (masked*padding_mask).mean(dim=0) # [BS, D]
        else:
            # contrative loss
            # reduce dimensionality, take sum across the sequence
            unmasked_rd = unmasked.mean(dim=0) # [BS, D]
            masked_rd = masked.mean(dim=0) # [BS, D]

        # normalize for cos similarity
        unmasked_rd_norm = F.normalize(unmasked_rd, dim=-1)
        masked_rd_norm = F.normalize(masked_rd, dim=-1)
        sim = torch.matmul(masked_rd_norm, unmasked_rd_norm.T) # [BS, BS]
        sim /= self.temp
        
        #####################################
        # similarity plot
        #
        # do not delete
        #
        #import matplotlib.pyplot as plt
        #plt.imshow(sim.detach().numpy())
        #plt.xticks([])
        #plt.yticks([])
        #plt.colorbar()
        #plt.show()
        #####################################

        # positive pairs are the diagonal elements, i.e. the same sequence in the two embeddings
        labels_batch_sim = torch.arange(sim.size(0), device=unmasked.device)
        loss_batch_sim = self.loss_fn(sim, labels_batch_sim)
        
        # if bs < number of targets, leads to an error in cross entropy
        if max(labels) > sim.size(0) and \
            (self.batch_size > sim.size(0)):
            # resample label
            unique = torch.unique(labels)
            mapping = {v.int().item(): i for i, v in enumerate(unique)}
            labels = torch.tensor([mapping[label.int().item()] for label in labels.clone()])
        try:
            # get the positive mask, positive pairs have the same class label
            pos_mask = labels.flatten().unsqueeze(0) == labels.flatten().unsqueeze(1)
            pos_mask = pos_mask.to(sim.device)
            #####################################
            # pos pair plot
            #
            # do not delete
            #
            #pos_mask_plot = pos_mask.int()
            #pos_mask_plot = torch.where(pos_mask_plot==0, -1, pos_mask_plot)
            #idx = torch.where(pos_mask_plot == 1)
            #for j, i in enumerate(idx[0]):
            #    label = labels.flatten()[i]
            #    pos_mask_plot[i, idx[1][j]] = label 
            #plt.imshow(pos_mask_plot.detach().numpy())
            #plt.xticks([])
            #plt.yticks([])
            #plt.show()
            #####################################
            # compute the log softmax, converts sim to probabilities (row sums to 1) and take logarithm (num. stability)
            log_probs = F.log_softmax(sim, dim=1)
            #
            #print(pos_mask.device, log_probs.device, labels.device)
            loss = -(pos_mask * log_probs).sum(dim=1) / pos_mask.sum(dim=-1)
            loss_class_sim = loss.mean()
            #if labels.dtype != torch.long: # convert float to double
            #    labels = labels.type(torch.long)
            #if len(labels.size()) >= 1:
            #    labels = labels.squeeze(1)
            #print(sim.size(), labels.size())
            #print(self.loss_fn)
            #loss_class_sim = self.loss_fn(sim, labels)
        except Exception as e:
            print(f'Warning from Contrastive Loss: {e}')
            print(sim.size(), labels.size())
            #loss_class_sim = deepcopy(self.loss_class_sim_prev)
            labels_batch_sim = torch.arange(sim.size(0), device=unmasked.device)
            loss_class_sim = self.loss_fn(sim, labels_batch_sim)
    #        loss_class_sim = self.loss_sim_prev

        self.loss_class_sim_prev = loss_class_sim.clone()
        
        loss = self.weight * loss_batch_sim + (1 - self.weight) * loss_class_sim
        return loss, loss_batch_sim, loss_class_sim


class DRMContrastiveLoss(nn.Module):
    def __init__(self, 
        temp: float = 0.5,
        weight_batch_class: float = 0.5,
        batch_size: int = None,
        pred_type: Literal['binary_cls', 'multiclass_cls']='binary_cls',
        weight_contrastive: float=1.0,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert batch_size is not None
        self._pred_type = pred_type
        self._weight_contrastive = weight_contrastive

        self.loss_contrstive = ContrastiveLoss(temp=temp, weight=weight_batch_class, batch_size=batch_size)

        if self._pred_type == 'multiclass_cls':
            self.ce = nn.CrossEntropyLoss()
        elif self._pred_type == 'binary_cls':
            self.ce = nn.BCEWithLogitsLoss()
        else:
            raise RuntimeError(f'{self._pred_type} is not implemented!')

    
    def forward(self, y_logits: Tensor, unmasked: Tensor, masked: Tensor, labels: Tensor, seq_len: Tensor=None, per_seq_element: bool=False):
        if self._pred_type == 'binary_cls':
            y_logits = y_logits.flatten()
            if unmasked.dtype == torch.double:
                labels = labels.type(unmasked.dtype)

            if labels.dtype == torch.long or labels.dtype == torch.int64:
                labels = labels.type(unmasked.dtype)
        #print(y_proba.dtype, labels.dtype) 
        if per_seq_element:
            sequential_labels = torch.concat([
                torch.concat([labels[i] for _ in range(seq_len[i])])
                for i in range(len(labels))
            ])
            loss_ce = self.ce(y_logits, sequential_labels.flatten())
        else:
            loss_ce = self.ce(y_logits, labels.flatten())
        
        
        loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_contrstive(
            unmasked=unmasked, masked=masked, labels=labels, seq_len=seq_len)
        #print(loss_ce.dtype, loss_contrastive.dtype)
        loss = loss_ce + self._weight_contrastive * loss_contrastive #+ loss_margin
        
        return loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim



class MBSLoss(nn.Module):
    """
    Margin Based Similarity Loss
    """
    def __init__(self, margin = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.margin = margin

    def forward(self, unmasked: Tensor, masked: Tensor, labels: Tensor):
        """
        Args:
            unmasked: unmasked embedding from encoder, [N, BS, D]
            masked: masked embedding from encoder, [N, BS, D]
            labels: ground truth labels, [N]
        """
        true_labels = labels.unsqueeze(0)
        pairwise_labels = (true_labels == true_labels.T)
        # create pos and neg labels
        pos_mask = pairwise_labels#.bool()
        neg_mask = ~pairwise_labels#.bool()
        
        # compute similarity
        # reduce dimensionality, take sum across the sequence
        unmasked_rd = unmasked.mean(dim=0) # [BS, D]
        masked_rd = masked.mean(dim=0) # [BS, D]
        # normalize for cos similarity
        unmasked_rd_norm = F.normalize(unmasked_rd, dim=-1)
        masked_rd_norm = F.normalize(masked_rd, dim=-1)
        sim = torch.matmul(masked_rd_norm, unmasked_rd_norm.T)
        # positive score
        pos_sim = sim[pos_mask]
        pos_loss = torch.mean(pos_sim**2)

        # negative score
        neg_sim = sim[neg_mask]
        neg_loss = torch.mean(torch.clamp(self.margin - neg_sim, min=0) ** 2)
        loss = pos_loss + neg_loss
        return loss
