import os.path as osp
import torch

from omegaconf import DictConfig

from .utils import initialize_activation_function, initialize_activation_function_from_str

try:
    from src.models import (
        LSTMNet, 
        STaRFormer
    )
    from src.util import TrainingMethodOptions
    from src.runtime import CentralizedModel

except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    from src.models import (
        LSTMNet, 
        STaRFormer
    )
    from src.util import TrainingMethodOptions
    from src.runtime import CentralizedModel


def initialize_lightning_model(config: DictConfig, **kwargs):
    """ initialize lightning module"""
    model = initialize_model(config=config, **kwargs)
    print(model)
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        lightning_model = CentralizedModel(config=config, model=model)
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        pass
    else:
        raise RuntimeError(f'{config.model.name} not found!')

    return lightning_model


def initialize_model(config: DictConfig, max_seq_len: int=None, dataset: object=None, **kwargs):
    """ Initialize Torch Model"""
    if config.model.name == 'lstm':
        model = lstm(config)
    elif config.model.name == 'starformer': 
        assert max_seq_len is not None, f'max_seq_len has to be given!'
        model = starformer(config, max_seq_len, dataset=dataset)
    else:
        raise RuntimeError(f'{config.model.name} not found!')

    return model


def lstm(config: DictConfig):
    return LSTMNet(
        input_size = config.model.input_size,
        hidden_size = config.model.hidden_size, 
        output_size = config.model.output_size, 
        num_layers = config.model.num_layers,
        dropout= config.model.dropout,
        batch_size = config.model.batch_size,
        device = config.model.device,
    )

def starformer(config: DictConfig, max_seq_len: int, dataset=None):
    activation = initialize_activation_function(config)
    activation_masking = initialize_activation_function_from_str(config.model.activation_masking)
    activation_cls = initialize_activation_function_from_str(config.model.activation_cls)

    if max_seq_len > config.model.embedding.max_seq_len:
        config.model.embedding.max_seq_len = max_seq_len
    
    return STaRFormer(
        ######## - - embedding - - ########
        # mts
        d_features=config.model.embedding.d_features,
        max_seq_len=config.model.embedding.max_seq_len,
        ######## - - embedding - - ########
        ######## - - transformer layers - - ########
        d_model=config.model.d_model,
        n_head=config.model.n_head,
        num_encoder_layers=config.model.num_encoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        activation=activation,
        layer_norm_eps=config.model.layer_norm_eps,
        batch_first=config.model.batch_first,
        bias=config.model.bias,
        enable_nested_tensor=config.model.enable_nested_tensor,
        mask_check=config.model.mask_check,
        masking=config.model.masking,
        mask_threshold=config.model.mask_threshold,
        mask_region_bound=config.model.mask_region_bound,
        ratio_highest_attention=config.model.ratio_highest_attention,
        aggregate_attn_per_batch=config.model.aggregate_attn_per_batch,
        activation_masking=activation_masking,
        activation_cls=activation_cls,
        return_attn=config.model.return_attn,
        ######## - - transformer layers - - ########
        ######## - - output head - - ########
        d_out=config.model.output_head.d_out,
        batch_size=config.model.output_head.batch_size,
        reduced=config.model.output_head.reduced,
        per_element_in_sequence_pred=config.model.output_head.per_element_in_sequence_pred,
        autoregressive_classification=config.model.output_head.autoregressive_classification,
        ######## - - output head - - ########
        precision=config.training.precision,
        reconstruction=config.model.output_head.reconstruction,
        
    )

        
