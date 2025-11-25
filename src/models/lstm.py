### OLD
import torch
import torch.nn as nn 

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from .output_heads_rnn import OutputHead, OutputHeadOptions

__all__ = ["LSTMNetOld", "LSTMNet"]


class LSTMNetOld(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float=0.0,
        batch_size: int=1,
        device: str="cpu"
        ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=False) # add an LSTM layer
        self.activation = nn.ReLU() # activation function after LSTM
        self.linear = nn.Linear(hidden_size, output_size) # fully-connected layer adter LSTM
        #self.activation_out = nn.Sigmoid() # activation function after Linear layer, before output

    def forward(self, x: PackedSequence) -> Tensor:
        assert isinstance(x, PackedSequence), f'Input x is not a {PackedSequence} but {type(x)}.'
        # x is a packed_seq
        device = x.data.device

        if device == "mps":
            x.data = x.data.type(torch.float32)
        # initial h0 and c0
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=x.data.dtype).to(device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=x.data.dtype).to(device)
        self.hidden = (h0, c0)
        #print(packed_seq.data.dtype, self.hidden[0].dtype, self.hidden[1].dtype)
        if x.data.dtype == torch.double:
            self.lstm.double()
            self.linear.double()
        #print(x.data.device, self.hidden[0].device, self.hidden[1].device) 
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        last_hidden_state = self.hidden[0][-1]
        activated_hidden = self.activation(last_hidden_state)
        out = self.linear(activated_hidden)
        #activated_out = self.activation_out(out)
        return out #activated_out

### New 
import torch
import torch.nn as nn

from typing import Literal, Callable, Union
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from .output_heads import OutputHead

class LSTMNet(nn.Module):
    def __init__(self, 
                 # RNN
                 input_size: int,
                 hidden_size: int, # num of cell states
                 num_layers: int=1,
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
                 *args, **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._task = task 
        self._batch_first = batch_first
        self._hidden_activation = nn.ReLU() # activation function after Rnn

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=False) # add an LSTM layer
        
        self.output_head = OutputHead(task=task, d_model=hidden_size, d_out=d_out,
                                      d_hidden=d_hidden, activation=activation, reduced=reduced, 
                                      cls_method=cls_method)

    def forward(self, x: Tensor | PackedSequence, N: Tensor=None, batch_size: int=None) -> Tensor:
        assert isinstance(x, PackedSequence) or isinstance(x, Tensor), f'Input x is not a {PackedSequence} but {type(x)}.'
        if batch_size is None:
            batch_size = x.data.size(0) if self._batch_first else x.data.size(1)
        # x is a packed_seq
        device = x.data.device
        dtype = x.data.dtype if x.data.dtype == torch.float32 else torch.float32
        if torch.backends.mps.is_available() and x.data.dtype == torch.double:
            x = x.to(dtype)
            self.lstm = self.lstm.to(device)
        
        # initial h0 and c0
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        c0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        self.hidden = (h0, c0)
        # lstm
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # get last hidden state for autoregressive prediction
        last_hidden_state = self.hidden[0][-1]
        activated_hidden = self._hidden_activation(last_hidden_state)
        # output head 
        if self._task == OutputHeadOptions.classification:
            logits = self.output_head(x=activated_hidden, N=N, batch_size=batch_size)
        else:
            raise NotImplementedError
        
        return logits
    