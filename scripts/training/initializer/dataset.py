import os.path as osp
import hydra
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict


try:
    from src.util import TrainingMethodOptions, DatasetOptions, DatasetPartitionOptions, ModelOptions
    from src.datamodule import (
        GeoLifeDatamodule,
        PAMDatamodule,
        UcrUeaDatamodule
    )

except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    from src.util import TrainingMethodOptions, DatasetOptions, DatasetPartitionOptions, ModelOptions
    from src.datamodule import (
        GeoLifeDatamodule,
        PAMDatamodule,
        UcrUeaDatamodule
    )
    

def initialize_datamodule(config: DictConfig):
    if config.dataset.lower() == DatasetOptions.geolife:
        datamodule, instance, config = geolife(config)
    elif config.dataset.lower() in DatasetOptions.ucr_uea:
        datamodule, instance, config = ucr_uea(config)
    elif config.dataset.lower() in DatasetOptions.pam:
        datamodule, instance, config = pam(config)
    else:
        raise RuntimeError(f'{config.dataset} is not found!')
    
    return datamodule, instance, config


def geolife(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        dm_params = {
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'train_test_split': config.datamodule.train_test_split,
            'identical_training_class_label_distribution': config.datamodule.identical_training_class_label_distribution,
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'pad_sequence': config.datamodule.pad_sequence,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,
            'max_trajectory_length': config.datamodule.max_trajectory_length,
            'synthetic_minority_upsampling': config.datamodule.synthetic_minority_upsampling,
            'noise_level': config.datamodule.noise_level,
        }

        datamodule = GeoLifeDatamodule(**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config


def ucr_uea(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        dm_params = {
            'dataset': config.datamodule.dataset,
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'train_splits': dict(config.datamodule.train_splits),
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'use_threads': config.datamodule.use_threads,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,
            
        }
        if config.dataset in [DatasetOptions.eigenworms, DatasetOptions.ethanolconcentration]:
            dm_params['max_trajectory_length'] = config.datamodule.max_trajectory_length

        datamodule = UcrUeaDatamodule(**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config

def pam(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        dm_params = {
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'train_split_index': config.datamodule.train_split_index,
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'use_threads': config.datamodule.use_threads,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,   
            'pad_sequence': config.datamodule.pad_sequence,
        }

        datamodule = PAMDatamodule(**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config


