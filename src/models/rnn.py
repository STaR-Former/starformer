import torch
import torch.nn as nn

from typing import Literal, Callable, Union
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from .output_heads_rnn import OutputHead, OutputHeadOptions


__all__ = ['RNNNet']


class RNNNet(nn.Module):
    def __init__(self, 
                 # RNN
                 input_size: int,
                 hidden_size: int, # num of cell states
                 num_layers: int=1,
                 nonlinearity: str='tanh',
                 bias: bool=True,
                 batch_first: bool=False,
                 dropout: float=0.0,
                 bidirectional: bool=False,
                 # Output head
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_out: int=1, 
                 d_hidden: int=None, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, 
                 cls_method: Literal['cls_token', 'autoregressive', 'elementwise']='autoregressive',
                 # Base Class
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._task = task 
        self._batch_first = batch_first
        self._hidden_activation = nn.ReLU() # activation function after Rnn

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, 
                          dropout=dropout, bidirectional=bidirectional)
        
        self.output_head = OutputHead(task=task, d_model=hidden_size, d_out=d_out,
                                      d_hidden=d_hidden, activation=activation, reduced=reduced, 
                                      cls_method=cls_method)
    
    def forward(self, x: Tensor | PackedSequence, N: Tensor=None, batch_size: int=None) -> Tensor:
        # RNN Hidden state versis RNN output
        # Hidden State:
        #   - Represents the final state of the RNN after processing the entire sequence.
        #   - Used in tasks requiring a summary representation of the entire sequence.
        #   - Commonly applied in classification tasks where the entire sequence is classified based on the final hidden state.
        # RNN Output:
        #   - Contains the output at each time step of the sequence.
        #   - Useful for tasks requiring information at each time step.
        #   - Applied in sequence-to-sequence tasks or when classifying each element in the sequence.
        if batch_size is None:
            batch_size = x.data.size(0) if self._batch_first else x.data.size(1)
        
        device = x.data.device
        dtype = x.data.dtype if x.data.dtype == torch.float32 else torch.float32
        if torch.backends.mps.is_available() and x.data.dtype == torch.double:
            x = x.to(dtype)
            self.rnn = self.rnn.to(device)

        # init hidden state
        self.hidden = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        # rnn
        out, hn = self.rnn(x, self.hidden)
        hn = self._hidden_activation(hn[-1])
        # output head 
        if self._task == OutputHeadOptions.classification:
            logits = self.output_head(x=hn)
        else:
            raise NotImplementedError
        return logits

