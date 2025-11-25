import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from PIL import Image

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset

from typing import Union, Callable, Optional, Any, List, Tuple, Literal
from copy import deepcopy
from src.util import EmbeddingData, DatasetOptions


class ClassificationHead(nn.Module):
    """
    ClassificationHead class for performing classification tasks.

    This class inherits from nn.Module and provides functionalities for linear transformation,
    aggregation, and applying sigmoid activation for classification.

    Attributes:
        linear (nn.Linear): Linear layer for transformation.
        aggregation (str): Aggregation method.
    """
    def __init__(self, d_model: int, d_out: int=1, aggregation: str = "mean"):
        """ 
        Args:
            d_model (int): Dimension of the input features.
            d_out (int, optional): Dimension of the output features. Defaults to 1.
            aggregation (str, optional): Aggregation method ('mean' or 'sum'). Defaults to "mean".
        """
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_out)
        self.aggregation = aggregation

    def forward(self, x: Union[Tensor, PackedSequence], padding_mask: Tensor, x_gaf: Tensor=None):
        if isinstance(x, PackedSequence):
            if x.data.dtype == torch.double:
                self.linear.double()
        else:
            if x.dtype == torch.double:
                self.linear.double()

        logits = self.linear(x)
        #logits_mask = logits.masked_fill(padding_mask.T.unsqueeze(-1), 0.0)
        logits_mask = logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.aggregation == 'mean':
            seq_lengths = (~padding_mask).sum(dim=1, keepdim=True)
            logits_aggr = logits_mask.sum(dim=0) / seq_lengths
        
        elif self.aggregation == 'sum':
            logits_aggr = logits_mask.sum(dim=0)
        if x_gaf is not None:
            # concatenate gaf fts and mts, mts-rbf fts
            logits_gaf = self.linear(x_gaf)
            logits_aggr = logits_aggr + logits_gaf 
        return logits_aggr
    

class PermuteBatchNorm(nn.Module):
    """
    PermuteBatchNorm class for permuting batch normalization.

    This class inherits from nn.Module and provides functionalities for permuting the dimensions
    of the input tensor for batch normalization.
    """
    def __init__(self, outhead: Literal['cls', 'masking']='cls', *args, **kwargs) -> None:
        """
        Args:
            outhead (Literal['cls', 'masking'], optional): Output head type. Defaults to 'cls'.
        """
        super().__init__(*args, **kwargs)
        self.outhead = outhead

    def forward(self, x):
        """Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Permuted tensor.
        """
        # x: [seq_len, bs, D] --> [bs, D, seq_len]
        if self.outhead == 'masking':
            return x.permute(1,2,0)
        if self.outhead == 'cls':
            return x.permute(1,0)


class InversePermuteBatchNorm(nn.Module):
    """
    InversePermuteBatchNorm class for inversely permuting batch normalization.

    This class inherits from nn.Module and provides functionalities for inversely permuting the dimensions
    of the input tensor for batch normalization.
    """
    def __init__(self, outhead: Literal['cls', 'masking']='cls', *args, **kwargs) -> None:
        """
        Args:
            outhead (Literal['cls', 'masking'], optional): Output head type. Defaults to 'cls'.
        """
        super().__init__(*args, **kwargs)
        self.outhead = outhead

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Inversely permuted tensor.
        """ 
        # x: [bs, D, seq_len] --> [seq_len, bs, D]
        if self.outhead == 'masking':
            return x.permute(2,0,1)
        if self.outhead == 'cls':
            return x.permute(1,0)

    
class ClassificationCLSTokenHead(nn.Module):
    """
    ClassificationCLSTokenHead class for performing classification using CLS token.

    This class inherits from nn.Module and provides functionalities for linear transformation,
    activation, and batch normalization for classification.
    """
    def __init__(self, 
        d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=False,
        ) -> None:
        """
        Args:
            d_model (int): Dimension of the input features.
            d_out (int, optional): Dimension of the output features. Default: 1.
            d_hidden (int, optional): Dimension of the hidden layer. Default: None.
            activation (Union[str, Callable[[Tensor], Tensor]], optional): Activation function. Default: nn.ReLU().
            reduced (bool, optional): Whether to use a reduced network. Default: False.
        """
        super().__init__()
        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                activation,
                nn.BatchNorm1d(d_hidden),
                nn.Linear(d_hidden, d_out),
            )

    def forward(self, x: Union[Tensor, PackedSequence], **kwargs):
        """
        Args:
            x (Union[Tensor, PackedSequence]): Input features.

        Returns:
            Tensor: Transformed features.
        """
        if isinstance(x, PackedSequence):
            if x.data.dtype == torch.double:
                self.net.double()
        else:
            if x.dtype == torch.double:
                self.net.double()
        
        if len(x.size()) >= 3:
            x = x[0, ...]
        else:
            x = x
        return self.net(x)
    

class ClassificationAutoregssiveHead(nn.Module):
    """
    ClassificationAutoregressiveHead class for performing autoregressive classification.

    This class inherits from nn.Module and provides functionalities for linear transformation,
    activation, and batch normalization for classification at the last point in the sequence.
    """
    def __init__(self, 
        d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=False,
        ) -> None:
        """
        Args:
            d_model (int): Dimension of the input features.
            d_out (int, optional): Dimension of the output features. Default: 1.
            d_hidden (int, optional): Dimension of the hidden layer. Default: None.
            activation (Union[str, Callable[[Tensor], Tensor]], optional): Activation function. Default: nn.ReLU().
            reduced (bool, optional): Whether to use a reduced network. Default: False.
        """
        super().__init__()
        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                activation,
                nn.BatchNorm1d(d_hidden),
                nn.Linear(d_hidden, d_out),
            )

    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor, batch_size: int): 
        """
        Args:
            x (Union[Tensor, PackedSequence]): Input features.
            N (Tensor): Tensor selecting the last element of the sequence.
            batch_size (int): Batch size.

        Returns:
            Tensor: Transformed features.
        """
        if isinstance(x, PackedSequence):
            if x.data.dtype == torch.double:
                self.net.double()
        else:
            if x.dtype == torch.double:
                self.net.double()
        # N selects the last element of the sequence, 
        # and the batch size select the correct element in the batch 
        # matching the seq_len selected
        x = x[N.flatten()-1, [i for i in range(batch_size)],...] # [batch_size, d_model]
        return self.net(x)
    

class Reshape(nn.Module):
    """
    Reshape class for reshaping tensors.

    This class inherits from nn.Module and provides functionalities for reshaping the input tensor.

    Args:
        *shape: Target shape for reshaping.
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class ClassificationPerSequentialElementHead(nn.Module):
    def __init__(self, 
        d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=False,
        batch_size: int=None
        ) -> None:
        super().__init__()
        self._d_out = d_out
        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
                #nn.Sigmoid()
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                activation,
                #Reshape(-1, d_hidden),
                #nn.BatchNorm1d(d_hidden),
                #Reshape(-1, batch_size, d_hidden),
                nn.Linear(d_hidden, d_out),
                #nn.Sigmoid()
            )

    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor, **kwargs):
        if isinstance(x, PackedSequence):
            if x.data.dtype == torch.double:
                self.net.double()
        else:
            if x.dtype == torch.double:
                self.net.double()
        logits = self.net(x) # shape [batch_size, seq_len, d_out] 
        logits = [logit[:N[idx].item(), :] for idx, logit in enumerate(logits)]    
        
        return torch.concat(logits, dim=0)
    

class MaskedModellingHead(nn.Module):
    def __init__(self, d_model, d_out, d_hidden: int=None, use_cls_token: bool=False, activation: Union[str, Callable[[Tensor], Tensor]] = nn.SiLU(), reduced: bool=True):
        super().__init__()

        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
                activation
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                activation,
                PermuteBatchNorm(outhead='masking'),
                nn.BatchNorm1d(d_hidden),
                InversePermuteBatchNorm(outhead='masking'),
                nn.Linear(d_hidden, d_out),
            )
        self.use_cls_token = use_cls_token

    def forward(self, x, padding_mask: Tensor, **kwargs):
        if isinstance(x, PackedSequence):
            if x.data.dtype == torch.double:
                self.net.double()
        else:
            if x.dtype == torch.double:
                self.net.double()

        if self.use_cls_token:
            x = x[1:, ...] # disregard cls token
            padding_mask = padding_mask[:, 1:]
        
        out = self.net(x)        
        out_masked = out.masked_fill(padding_mask.T[..., None], 0.0)
        return out_masked
