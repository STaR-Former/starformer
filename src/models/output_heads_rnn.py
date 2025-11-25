import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import Union, Callable, Literal


class OutputHead(nn.Module):
    def __init__(self, 
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_model: int=None, d_out: int=1, d_hidden: int=None, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, cls_method: Literal['cls_token', 'autoregressive', 'elementwise']='autoregressive',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._options = OutputHeadOptions()
        if task == self._options.classification:
            assert d_model is not None and isinstance(d_model, int)
            assert d_out is not None and isinstance(d_out, int)
            self.net = ClassificationHead(d_model=d_model, d_out=d_out, d_hidden=d_hidden,
                                          activation=activation, reduced=reduced, cls_method=cls_method)
        elif task == self._options.regression:
            # Implement regression head initialization
            raise NotImplementedError
        elif task == self._options.forecasting:
            # Implement forecasting head initialization
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported task: {task}")

    
    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        return self.net(x=x, N=N, batch_size=batch_size)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=True, cls_method: Literal['cls_token', 'autoregressive', 'elementwise']='autoregressive',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cls_options = ClassificationOptions()
        self._cls_method = cls_method

        # always return logits and not probas
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
                nn.BatchNorm1d(d_hidden),
                nn.Linear(d_hidden, d_out),
                #nn.Sigmoid()
            )
    
    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        if self._cls_method == self._cls_options.cls_token:
            return self._forward_cls_token(x=x)
        elif self._cls_method == self._cls_options.autoregressive:
            return self._forward_autoregessive(x=x, N=N, batch_size=batch_size)
        elif self._cls_method == self._cls_options.cls_token:
            assert N is not None, f'N cannot be `None`'
            return self._forward_elementwise(x=x, N=N)
        else:
            raise ValueError(f'{self._cls_method} is not known, possible options are {self._cls_options.cls_token}')

    def _forward_cls_token(self, x: Union[Tensor, PackedSequence]) -> Tensor:
        if len(x.size()) >= 3:
            x = x[0, ...]
        else:
            x = x
        return self.net(x)

    def _forward_autoregessive(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        # x [N, batch_size, d_model]
        # N selects the last element of the sequence, 
        # and the batch size select the correct element in the batch 
        # matching the seq_len selected
        #x = x[N.flatten()-1, [i for i in range(batch_size)],...] # [batch_size, d_model]
        if len(x.size()) == 3:
            assert N is not None, f'N cannot be `None`'
            assert batch_size is not None, f'batch_size cannot be `None`'
            x = x[N.flatten()-1, list(range(batch_size)),...] # [batch_size, d_model] (ensure correct last element is selected, even for padded sequences)
        elif len(x.size()) == 2: # for rnns (use last hidden state)
            x = x # [batch_size, d_model]
        return self.net(x)

    def _forward_elementwise(self, x: Union[Tensor, PackedSequence], N: Tensor, **kwargs) -> Tensor:
        print('check this implementation')
        logits = self.net(x) # shape [batch_size, seq_len, d_out] 
        logits = [logit[:N[idx].item(), :] for idx, logit in enumerate(logits)]    
        return torch.concat(logits, dim=0)



class OptionsBaseClass:
    @classmethod
    def get_options(cls):
        #return {key: value for key, value in cls.__dict__.items() if not key.startswith('__')}
        return [value for key, value in cls.__dict__.items() if not key.startswith('__') and key != 'get_options']


class ClassificationOptions(OptionsBaseClass):
    cls_token: str='cls_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise' # per element in sequence
    

class OutputHeadOptions(OptionsBaseClass):
    classification: str='classification'
    regression: str='regression'
    forecasting: str='forecasting'