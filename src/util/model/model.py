from torch import Tensor
from ..dataset import BaseData
from torch.nn.utils.rnn import PackedSequence

_all_ = [
    'EmbeddingData'
]

class EmbeddingData(BaseData):
    def __init__(
        self,
        temporal: Tensor | PackedSequence=None,
        spatial: Tensor | PackedSequence=None,
        radial: Tensor | PackedSequence=None,
        **kwargs
        ) -> None:
        self.temporal = temporal
        self.spatial = spatial
        self.radial = radial
        super().__init__(**kwargs)