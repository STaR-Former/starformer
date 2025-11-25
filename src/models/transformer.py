import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from typing import Union, Callable, Optional, List, Tuple, Literal
from src.util import  DatasetOptions, MaskingOptions

import src.nn as src_nn

from .output_heads import (
    ClassificationCLSTokenHead, ClassificationPerSequentialElementHead, 
    ClassificationAutoregssiveHead, MaskedModellingHead
)


__all__ = ["STaRFormer"]


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding class for adding positional information to the input embeddings.

    This class inherits from nn.Module and provides functionalities to add positional encodings
    to the input tensor, which helps the model to understand the order of the sequence.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): Positional encoding tensor.
    """
    def __init__(self, 
        d_model: int, 
        dropout: float = 0.1, 
        max_seq_len: int = 5000, 
        *args, **kwargs) -> None:
        """
        Args:
            d_model (int): Dimension of the model.
            dropout (float, optional): Dropout rate. Default: 0.1.
            max_seq_len (int, optional): Maximum sequence length. Default: 5000.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) # selects every even element
        pe[:, 0, 1::2] = torch.cos(position * div_term) # selects ever odd element
        self.register_buffer('pe', pe)

    def forward(self, x: Union[Tensor, PackedSequence], batch_first: bool=False) -> Tensor:
        """
        Args:
            x (Union[Tensor, PackedSequence]): Input tensor of shape [seq_len, batch_size, embedding_dim].
            batch_first (bool, optional): If True, the input tensor shape is [batch_size, seq_len, embedding_dim]. Default: False.

        Returns:
            Tensor: Tensor with added positional encoding.
        """
        if batch_first:
            x = self.pe[:x.size(1)].permute(1,0,2)
        else:
            x =  self.pe[:x.size(0)]
        return self.dropout(x)


class STaRFormer(nn.Module):
    """
    Implementation of `STaRFormer: Semi-Supervised Task-Informed Representation Learning 
    via Attention-Based Dynamical Regional Masking for Transformers in Sequential Data`.

    This class inherits from nn.Module and provides functionalities for embedding, 
    transformer layers, and output heads for various tasks such as classification, 
    reconstruction, and autoregressive classification.
    """
    def __init__(self, 
        ### embedding
        d_features: int = 9,
        max_seq_len: int = 487, # max in data: 487
        ### embedding
        ### transformer layer params
        d_model: int = 16, #512, 
        n_head: int = 1, #8, 
        num_encoder_layers: int = 1, #6,
        dim_feedforward: int = 16, #2048, 
        dropout: float = 0.0, #.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.silu,
        layer_norm_eps: float = 1e-5, 
        batch_first: bool = False, 
        bias: bool = True, 
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        masking: Literal['random', 'drm'] = None,
        mask_threshold: float=None,
        mask_region_bound: float=0.1,
        ratio_highest_attention: float=0.5,
        aggregate_attn_per_batch: bool=False,
        activation_masking: Union[str, Callable[[Tensor], Tensor]] = nn.SiLU(),
        activation_cls: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        return_attn: bool = False,
        ### transformer layer params
        device=None, 
        dtype=None,
        # output head params
        d_out: int = 1,
        batch_size: int=16,
        reduced: bool=False,
        precision: Union[str, int]=32,
        reconstruction: bool=False,
        per_element_in_sequence_pred: bool=False,
        autoregressive_classification: bool=False,
        *args, **kwargs
        ) -> None:
        """
        Initializes STaRFormer

        Args:
            d_features (int, optional): Dimension of the input features. Default: 9.
            max_seq_len (int, optional): Maximum sequence length. Default: 487.
            d_model (int, optional): Dimension of the model. Default: 16.
            n_head (int, optional): Number of attention heads. Default: 1.
            num_encoder_layers (int, optional): Number of encoder layers. Default: 1.
            dim_feedforward (int, optional): Dimension of the feedforward network. Default: 16.
            dropout (float, optional): Dropout rate. Default: 0.0.
            activation (Union[str, Callable[[Tensor], Tensor]], optional): Activation function. Default: F.silu.
            layer_norm_eps (float, optional): Epsilon value for layer normalization. Default: 1e-5.
            batch_first (bool, optional): Whether the batch dimension is the first dimension. Default: False.
            bias (bool, optional): Whether to use bias in linear layers. Default: True.
            norm (Optional[nn.Module], optional): Normalization layer. Default: None.
            enable_nested_tensor (bool, optional): Whether to enable nested tensor. Default: True.
            mask_check (bool, optional): Whether to check masking. Default: True.
            masking (Literal['random', 'darem'], optional): Masking strategy. Default: None.
            mask_threshold (float, optional): Threshold for masking. Default: None.
            mask_region_bound (float, optional): Region bound for masking. Default: 0.1.
            ratio_highest_attention (float, optional): Ratio of highest attention. Default: 0.5.
            aggregate_attn_per_batch (bool, optional): Whether to aggregate attention per batch. Default: False.
            activation_masking (Union[str, Callable[[Tensor], Tensor]], optional): Activation function for masking. Default: nn.SiLU().
            activation_cls (Union[str, Callable[[Tensor], Tensor]], optional): Activation function for classification. Default: nn.ReLU().
            return_attn (bool, optional): Whether to return attention weights. Default: False.
            device (optional): Device to run the model on. Default: None.
            dtype (optional): Data type for the model. Default: None.
            d_out (int, optional): Dimension of the output features. Default: 1.
            batch_size (int, optional): Batch size. Default: 16.
            reduced (bool, optional): Whether to use a reduced network. Default: False.
            precision (Union[str, int], optional): Precision for the model. Default: 32.
            reconstruction (bool, optional): Whether to perform reconstruction. Default: False.
            per_element_in_sequence_pred (bool, optional): Whether to perform per-element prediction in sequence. Default: False.
            autoregressive_classification (bool, optional): Whether to perform autoregressive classification. Default: False.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first
        self.masking = masking
        self.return_attn = True if masking is not None and masking == 'drm' else False
        self.aggregate_attn_per_batch = aggregate_attn_per_batch
        self.precision = precision
        self.reconstruction = reconstruction
        self._per_element_in_sequence_pred = per_element_in_sequence_pred
        self._autoregressive_classification = autoregressive_classification

        #self.return_attn = return_attn
        if self.masking is not None:    
            assert masking in ['random', 'drm', None], f'{masking} is not available, use "random", "drm" or "None"!'
            if masking == MaskingOptions.drm:
                assert mask_region_bound is not None, f'{mask_region_bound} cannot be None!'
                assert ratio_highest_attention is not None, f'{ratio_highest_attention} cannot be None!'

            assert mask_threshold is not None, f'{mask_threshold} cannot be None!'
            
            self.mask_threshold = mask_threshold
            self._attn_weights = None
            self.mask_region_bound = mask_region_bound
            self.ratio_highest_attention = ratio_highest_attention

        self.encoder = self._build_encoder(
            d_model=d_model, n_head=n_head, num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            layer_norm_eps=layer_norm_eps, batch_first=batch_first, bias=bias,
            norm=norm, enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check, return_attn=self.return_attn,
        )

        # mts embedding
        self.linear_emb = nn.Linear(d_features, d_model)
        self.linear_emb_activation = nn.Tanh()
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_seq_len=max_seq_len)
        
        # output head 
        if self._per_element_in_sequence_pred and self._autoregressive_classification:
            raise RuntimeError(f'Conflicting configuration: per_element_in_sequence_pred={self._per_element_in_sequence_pred} \
and autoregressive_classification={self._autoregressive_classification}, only one can be `True`')
        elif self._per_element_in_sequence_pred:
            self.output_head = ClassificationPerSequentialElementHead(d_model=d_model, d_out=d_out, activation=activation_cls, reduced=reduced, batch_size=batch_size)
        elif self._autoregressive_classification:
            self.output_head = ClassificationAutoregssiveHead(d_model=d_model, d_out=d_out, activation=activation_cls, reduced=reduced)
        else: # default
            self.cls_token = nn.Parameter(torch.zeros(1, batch_size, d_model)) if not batch_first else nn.Parameter(torch.zeros(batch_size, 1, d_model))
            self.output_head = ClassificationCLSTokenHead(d_model=d_model, d_out=d_out, activation=activation_cls, reduced=reduced)

        if self.masking is not None and self.reconstruction:
            self.masked_modelling_head = MaskedModellingHead(d_model, d_features, use_cls_token=True, activation=activation_masking, reduced=reduced)

    def _build_encoder(self, 
        d_model: int,
        n_head: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: Union[str, Callable[[Tensor], Tensor]],
        layer_norm_eps: float,
        batch_first: bool,
        bias: bool,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        return_attn: bool = True,
        #need_weights: bool = False,
        **kwargs):
        return src_nn.TransformerEncoder(
            d_model=d_model,
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=activation, 
            layer_norm_eps=layer_norm_eps, 
            batch_first=batch_first, 
            bias=bias,
            num_layers=num_encoder_layers,
            norm=norm,
            enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check,
            return_attn=return_attn,
            #need_weights=need_weights,
            **kwargs
        )
        #return encoder #nn.ModuleDict({k: deepcopy(encoder) for k in ['encoder']})

    def forward(self, x: Tensor, N: Tensor, padding_mask: Tensor=None, mode: Literal['train', 'val', 'test']='train', dataset: str=None) -> Tensor:
        """
        Forward pass for the STaRFormer model.

        Args:
            x (Tensor): Input sequential data of shape [seq_len, bs, D].
            N (Tensor): Sequence lengths of each element in the batch.
            padding_mask (Tensor, optional): Mask for padded sequential input data. Default: None.
            mode (Literal['train', 'val', 'test'], optional): Mode of operation. Default: 'train'.
            dataset (str, optional): Dataset name. Default: None.

        Returns:
            Tensor: Output logits or a dictionary containing various outputs based on the mode and masking.
        """
        assert isinstance(x, Tensor), "Input x must be a tensor."
        # Set precision
        if self.precision == 32:
            x.data = x.data.type(torch.float32)
        elif self.precision == 64:
            if x.data.dtype != torch.double:
                x.data = x.data.type(torch.double)
            self.linear_emb.double()
            self.encoder.double()
            
        # Move data to the appropriate device
        keys = list(self.state_dict().keys())
        device = self.state_dict()[keys[-1]].device
        if x.data.device != device:
            x = x.to(device)

        # Create padding mask if not provided
        padding_mask = (x.data.sum(dim=-1) == 0).T if padding_mask is None else padding_mask

        # embedding
        x_emb = self.linear_emb_activation(self.linear_emb(x.data))
        bs = x.size(0) if self.batch_first else x.size(1)
        if not self._per_element_in_sequence_pred and not self._autoregressive_classification:
            x_emb = torch.concat([self.cls_token[:, :bs, :], x_emb], dim=0)
            # add cls token dim padding mask
            # here added as false so encluded in attention, can also use True to not include in attention
            padding_mask = torch.concat([torch.tensor([False]*bs, device=padding_mask.device).reshape(-1,1), padding_mask], dim=1)
        x_emb = x_emb + self.pos_encoder(x_emb, batch_first=False)
        
        padding_mask_out = None
        sequence_mask = None

        ###############
        # transformer #
        ###############
        if self.masking == MaskingOptions.drm:
            # darem is a synonym for darem, not consistent in code base
            # ensure return attn_weights is activated
            for layer in self.encoder.transformer_encoder.layers:
                if layer.return_attn is False: 
                    layer.return_attn = True

        if dataset in DatasetOptions.ucr_uea:
            embedding_cls, attn_weights = self.encoder.forward(src=x_emb, N=N, mask=None, 
                src_key_padding_mask=None, aggregate_attn_per_batch=self.aggregate_attn_per_batch)
        else:
            embedding_cls, attn_weights = self.encoder.forward(src=x_emb, N=N, src_key_padding_mask=padding_mask, aggregate_attn_per_batch=self.aggregate_attn_per_batch)

        if self.masking is not None: # and mode in ['train', 'val']:
            padding_mask_out = x[:, :, :] == 0.0
            if self.masking == MaskingOptions.random:
                x_emb_masked, sequence_mask = random_batchwise_masking(x=x_emb.clone(), N=N, mask_p=self.mask_threshold)
            
            elif self.masking == MaskingOptions.drm:
                if self.encoder.transformer_encoder.layers[0].self_attn_weights is None or \
                    attn_weights is None:
                    # random sampling for initial masking
                    x_emb_masked, sequence_mask = random_batchwise_masking(x=x_emb.clone(), N=N, mask_p=self.mask_threshold)
                else:
                    # DAReM
                    x_emb_masked, sequence_mask = dynmaic_regional_batchwise_masking(x=x_emb.clone(), N=N, attn_matrices=attn_weights,
                        mask_region_bound=self.mask_region_bound, mask_threshold=self.mask_threshold, 
                        ratio_highest_attention=self.ratio_highest_attention, batch_first=self.batch_first
                    )
            else:
                raise RuntimeError(f'{self.masking} not given or known!')

            if self.masking == MaskingOptions.drm:
                # ensure return attn_weights is deactivtae for more efficient calculation
                for layer in self.encoder.transformer_encoder.layers:
                    if layer.return_attn is True: 
                        layer.return_attn = False
            
            if dataset in DatasetOptions.ucr_uea:
                embedding_masked, _ = self.encoder.forward(src=x_emb_masked, N=N, mask=None,
                    src_key_padding_mask=None, aggregate_attn_per_batch=self.aggregate_attn_per_batch)
            else:
                embedding_masked, _ = self.encoder.forward(src=x_emb_masked, N=N, src_key_padding_mask=padding_mask, aggregate_attn_per_batch=self.aggregate_attn_per_batch)
       
        if self.masking == MaskingOptions.drm:
            self._attn_weights = attn_weights

        # output head
        if self.masking is not None and mode in ['train', 'val'] and self.reconstruction:
            out_representation = self.masked_modelling_head(embedding_masked, padding_mask=padding_mask)
        
        if self._per_element_in_sequence_pred:
            logits = self.output_head(**{
                'x': embedding_cls.permute(1, 0, 2), # change (batch_size, seq_len, feature_dim)
                'N': N,
            })
        elif self._autoregressive_classification:
            logits = self.output_head(**{
                'x': embedding_cls.permute(1, 0, 2), # change (batch_size, seq_len, feature_dim)
                'N': N,
                'batch_size': bs,
            })
        else:
            logits = self.output_head(**{
                'x': embedding_cls, 'padding_mask': padding_mask, 'x_gaf': None
            })

        # output
        out_dict = {'logits': logits,}
        
        if self.masking is not None and mode in ['train', 'val']:
            # add repeat for d_model == 8 to have enough dimnesinality[1:, :, :padding_mask_out.size(-1)].size())
            repeat_factor = int(padding_mask_out.size(-1) / sequence_mask.size(-1)) + 1
            out_dict['embedding_cls'] = embedding_cls
            out_dict['embedding_masked'] = embedding_masked
            if self.reconstruction: out_dict['out_representation']= out_representation
            out_dict['padding_mask']= padding_mask_out 
            out_dict['sequence_mask']= sequence_mask.repeat(1,1,repeat_factor)[1:, :, :padding_mask_out.size(-1)]

            #for key, values in out_dict.items():
            #    print(key, values.dtype)
            return out_dict 
        else:
            out_dict['embedding_cls'] = embedding_cls
            #out_dict['embedding_masked'] = embedding_masked
            return out_dict


def random_batchwise_masking(x: Tensor, N: Tensor, mask_p=0.2, batch_first: bool=False) -> Tuple[Tensor, Tensor]:
    # create indices which will be masked
    mask_indices = [
        torch.randint(0, n.item(), (int(n.item()*mask_p),)).sort().values if int(mask_p) != 1 else torch.arange(0, n.item(),1)
        for i, n in enumerate(N)
    ]
    
    # apply mask
    masked_batch = x # torch.zeros_like(x, dtype=x.dtype, device=x.device)
    for i, indices in enumerate(mask_indices):
        if not batch_first:
            masked_batch_item = masked_batch[:, i, :]
            # create mask
            masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
            masked_batch[:, i, :] = masked_batch_item

    # test 
    for i, indices in enumerate(mask_indices):
        assert torch.all(masked_batch[indices, i, :]==0.0)
    
    sequence_mask = torch.zeros_like(x, dtype=torch.bool)
    for i, m in enumerate(mask_indices):
        sequence_mask[m, i, :] = True
    return masked_batch, sequence_mask

def attention_rollout(attn_matrices: List[Tensor], seq_len: Tensor=None) -> Tensor:
    """
    https://arxiv.org/pdf/2005.00928
    attention includes the special token
    """
    # rollout 0
    rollout = attn_matrices[0]
    bs = rollout.size(0)
    max_len = rollout.size(1)

    # Create a mask
    if seq_len is None: # default
        padding_mask = torch.ones(bs, max_len, max_len, device=rollout.device)
    else:
        # actual mask
        padding_mask = torch.zeros(bs, max_len, max_len, device=rollout.device)
    
        # Fill the mask tensor with 1s for valid positions and 0s for padding
        #print(bs, seq_len.size())
        for i in range(bs):
            if i <= (seq_len.size(0)-1): # in case batch size and attn don't match
                padding_mask[i, :seq_len[i].item()+1, :seq_len[i].item()+1] = 1.0 # add plus 1 for special token
    
    rollout = rollout*padding_mask
    for A in attn_matrices[1:]:
        rollout = torch.matmul(
            0.5*A*padding_mask + 0.5*torch.eye(A.shape[1], device=A.device),
            rollout
        )

    return rollout

def compute_attention_scores(attn_matrices: List[Tensor], seq_len: Tensor=None):
    rollout = attention_rollout(attn_matrices, seq_len=seq_len) 
    if len(rollout.size()) == 2: # per batch
        diag = torch.diagonal(rollout, offset=0)
        attn_scores = rollout.sum(dim=0) - diag
    elif len(rollout.size()) == 3: # per element in batch
        diag = torch.diagonal(rollout, offset=0, dim1=1, dim2=2)
        attn_scores = rollout.sum(dim=1) - diag
    
    return attn_scores

def dynmaic_regional_batchwise_masking(
    x: Tensor, 
    N: Tensor,
    attn_matrices: List[Tensor], 
    mask_region_bound: float=0.1,
    mask_threshold=0.2,
    ratio_highest_attention: float = 0.5,
    batch_first: bool=False,
    verbose: bool=False,
    ) -> Tuple[Tensor, Tensor]:
    # compute attention scores via attentino rollout
    #for attn_matrix in attn_matrices:
    #    print('matrix', attn_matrix.size())
    #attn_scores = compute_attention_scores(attn_matrices)
    attn_scores = compute_attention_scores(attn_matrices, seq_len=N)
    bs = x.size(0) if batch_first else x.size(1)
    #print('\nregionalmasking')
    #print(attn_scores.size(), x.size(), N, '\n')
    # find top k features
    if len(attn_scores.size()) == 1 or attn_scores.size(0) != bs: # per batch, check for issues with batch size not matching, lightning issue with sanity check
        #print('darem per batch')
        if attn_scores.size(1) != bs and len(attn_scores.size()) != 1:
            attn_scores = attn_scores.mean(dim=0) # aggr per batch
        if any(elem != 25 for elem in N):
            # found at least one element that is padded 
            mask_indices = [
                _dynmaic_regional_masking(
                    x=x, N=N, attn_scores=attn_scores, batch_first=batch_first,
                    ratio_highest_attention=ratio_highest_attention, mask_region_bound=mask_region_bound, 
                    mask_threshold=mask_threshold, verbose=verbose,
                )
                for batch_idx, n in enumerate(N)
            ]
            # apply mask batchwise
            x_batch_masked = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            for i, indices in enumerate(mask_indices):
                if not batch_first:
                    #masked_batch_item = deepcopy(x[:, i, :]) 
                    masked_batch_item = x[:, i, :]
                    # create mask
                    masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
                    x_batch_masked[:, i, :] = masked_batch_item
                else:
                    #masked_batch_item = deepcopy(x[i, ...]) 
                    masked_batch_item = x[i, ...]
                    # create mask
                    masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
                    x_batch_masked[i, ...] = masked_batch_item

        else:
            mask_indices = _dynmaic_regional_masking(
                    x=x, N=N, attn_scores=attn_scores, batch_first=batch_first,
                    ratio_highest_attention=ratio_highest_attention, mask_region_bound=mask_region_bound, 
                    mask_threshold=mask_threshold, verbose=verbose,
                )

            x_batch_masked = x
            if not batch_first:
                x_batch_masked[mask_indices, ...] = x_batch_masked[mask_indices, ...] = 0.0  
            else:
                x_batch_masked[:, mask_indices, ...] = x_batch_masked[:, mask_indices, ...] = 0.0
            
            mask_indices = [mask_indices for _ in range(len(N))]
        
    elif len(attn_scores.size()) == 2: # per element in batch
        #print('darem batchwise')
        mask_indices = _dynmaic_regional_masking_per_element_in_batch(
            x=x, N=N, attn_scores=attn_scores, batch_first=batch_first,
            ratio_highest_attention=ratio_highest_attention, mask_region_bound=mask_region_bound, 
            mask_threshold=mask_threshold, verbose=verbose,
        )
        # apply mask batchwise
        x_batch_masked = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for i, indices in enumerate(mask_indices):
            if not batch_first:
                masked_batch_item = x[:, i, :]
                # create mask
                masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
                x_batch_masked[:, i, :] = masked_batch_item
            else:
                masked_batch_item = x[i, ...]
                # create mask
                masked_batch_item[indices, ...] = masked_batch_item[indices, ...] = 0.0
                x_batch_masked[i, ...] = masked_batch_item

    
    sequence_mask = torch.zeros_like(x, dtype=torch.bool)
    for i, m in enumerate(mask_indices):
        sequence_mask[m, i, :] = True

    return x_batch_masked, sequence_mask


def _dynmaic_regional_masking(
        x: Tensor,
        N: Tensor,
        attn_scores: Tensor,
        batch_first: bool,
        ratio_highest_attention: float,
        mask_region_bound: float,
        mask_threshold: float,
        verbose: bool=False
    ) -> List[int]:
    seq_len = x.size(1) if batch_first else x.size(0)
    top_k_values, top_k_indices = attn_scores.topk(int(math.ceil(
        ratio_highest_attention * seq_len)))
    
    try:
        top_index = top_k_indices[0].detach().item()
    except:
        #print(top_index)
        raise RuntimeError

    top_index_region_bounds = int(seq_len*mask_region_bound) 
    
    mask_indices = list(
        range(max(0, top_index - top_index_region_bounds), min(seq_len, top_index + top_index_region_bounds + 1))
    )
    if verbose: print('created mask for top feature')
    if len(mask_indices) <= int(seq_len*mask_threshold):
        for i, top_k_index in enumerate(top_k_indices[1:]):
            if len(mask_indices) <= int(seq_len*mask_threshold):
                top_k_index_region_bounds = int(seq_len*mask_region_bound)
                mask_indices_i = list(
                    range(max(0, top_k_index - top_k_index_region_bounds), min(seq_len, top_k_index + top_k_index_region_bounds + 1))
                )
                
                mask_indices.extend(mask_indices_i)
                if verbose: print(f'created mask for top {i+1} feature')

    # fill up with random
    mask_indices = sorted(list(set(mask_indices)))

    for i, n in enumerate(N):
        if len(mask_indices) < int(n.item()*mask_threshold):
            available_idx = list(set(range(n.item())) - set(mask_indices))
            n_diff = int((n.item()*mask_threshold) - len(mask_indices))
            selected = random.sample(available_idx, n_diff)
            mask_indices.extend(selected)
    
    mask_indices = sorted(list(set(mask_indices)))

    return mask_indices


def _dynmaic_regional_masking_per_element_in_batch(
        x: Tensor,
        N: Tensor,
        attn_scores: Tensor,
        batch_first: bool,
        ratio_highest_attention: float,
        mask_region_bound: float,
        mask_threshold: float,
        verbose: bool=False
    ) -> List[int]:
    
    seq_len = x.size(1) if batch_first else x.size(0)
    bs = x.size(0) if batch_first else x.size(1)
    #print('attn_scores darem', attn_scores.size(), N.size(), seq_len, bs)
    top_k_indices = [
        attn_scores[i].topk(int(math.ceil(ratio_highest_attention * n.item())), dim=0)
        for i, n in enumerate(N)
    ]
    
    #print(seq_len, x.size(0), top_k_indices, top_k_indices[0][0])
    mask_indices = []
    for elem in range(bs):
        top_index = top_k_indices[elem][1][0].detach().item()
        #print('top_index', top_index)
        #print('mask_region_bound', mask_region_bound)
        top_index_region_bounds = int(N[elem].item()*mask_region_bound) 
        #print(top_index_region_bounds)
        #print(top_index, top_index_region_bounds)
        mask_indices_i = list(
            range(max(0, top_index - top_index_region_bounds), min(N[elem].item(), top_index + top_index_region_bounds + 1))
        )
        #print(range(max(0, top_index - top_index_region_bounds), min(N[elem].item(), top_index + top_index_region_bounds + 1)))
        #print(mask_indices_i)
        #print('mask_indices', mask_indices)
        if verbose: print('created mask for top feature')
        #print(len(mask_indices_i), int(seq_len*mask_threshold))
        if len(mask_indices_i) <= int(N[elem].item()*mask_threshold):
            for i, top_k_index in enumerate(top_k_indices[elem][1][1:]):
                if len(mask_indices_i) <= int(N[elem].item()*mask_threshold):
                    top_k_index_region_bounds = int(N[elem].item()*mask_region_bound)
                    #print(top_k_index.item(), top_k_index_region_bounds, seq_len, top_k_index + top_k_index_region_bounds + 1)
                    mask_indices_j = list(
                        range(max(0, top_k_index.item() - top_k_index_region_bounds), min(N[elem].item(), top_k_index.item() + top_k_index_region_bounds + 1))
                    )
                    
                    mask_indices_i.extend(mask_indices_j)
                    if verbose: print(f'created mask for top {i+1} feature')
        # fill up with random
        mask_indices_i = sorted(list(set(mask_indices_i)))
        
        # add random index if necessary
        if len(mask_indices_i) < int(N[elem].item()*mask_threshold):
            available_idx = list(set(range(N[elem].item())) - set(mask_indices_i))
            n_diff = int((N[elem].item()*mask_threshold) - len(mask_indices_i))
            selected = random.sample(available_idx, n_diff)
            mask_indices_i.extend(selected)

        mask_indices_i = sorted(list(set(mask_indices_i)))
        mask_indices.append(mask_indices_i)

    # for mask visualization
    # do not delete
    #import matplotlib.pyplot as plt
    #import numpy as np

    #N, bs, _ = x.size()

    #plot_Arr = np.zeros((N, bs))
    #for i, m_elem in enumerate(mask_indices):
        #print(plot_Arr.shape)
    #    plot_Arr[mask_indices[i], i] = 1.0
        #print(i, len(m_elem), m_elem)

    #np.save(f"../results/mask_pam_b7czzsgq_{mask_region_bound}.npy", plot_Arr)
    ##print(np.unique(plot_Arr, return_counts=True))
    #fig = plt.figure(figsize=(10,20))
    #plt.imshow(plot_Arr.T, cmap='viridis')
    #plt.colorbar()
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()

    return mask_indices

    

    
    






