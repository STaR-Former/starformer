import os
import shutil
import requests
import logging
import zipfile

import torch
from torch import Tensor
from pathlib import Path
from typing import List, Union
from tqdm.auto import tqdm

from src.util.dataset.utils import BaseData
from src.util import TrainingMethodOptions
from torch.utils.data import Dataset

import numpy as np


__all__ = [
    'PAMData',
    'PAMBatchData'
]


class PAMData(BaseData):
    def __init__(
        self,
        seq_id: str=None, 
        data: Tensor=None, 
        label: Tensor=None,
        **kwargs
        ) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class PAMBatchData(BaseData):
    def __init__(
        self,
        seq_id: List[str]=None,
        data: Tensor=None, 
        label: Tensor=None, 
        seq_len: List |  Tensor=None,
        **kwargs
        ) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.batch_size=len(label)
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0])]
        for idx, sl in enumerate(self.seq_len):
            self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        super().__init__(**kwargs)

############
# download #
############

def download_pam(
    url: Union[str, Path, os.PathLike[str]], 
    folder_path: Union[str, Path, os.PathLike[str]],
    file_name: str,
    logger: logging.Logger=None,
    ):

    # Make sure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Download the file in chunks and write to the file directly
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        zip_file_path = os.path.join(folder_path, file_name)
        _download(response=response, zip_file_path=zip_file_path, logger=logger)
        unzip(folder_path=folder_path, zip_file_path=zip_file_path, logger=logger)
        organize(folder_path=folder_path, logger=logger)
    else:
        raise DownloadError(f"Failed to download file. HTTP Status code: {response.status_code}")

def _download(
    response: requests.Response,
    zip_file_path: Union[str, Path, os.PathLike[str]], 
    logger: logging.Logger=None,
    ):
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 Kibibytes
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc='Donwloading PAM')

    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                t.update(len(chunk))
    t.close()
    
    msg = f"ZIP file downloaded successfully and saved to {zip_file_path}."
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def unzip(
    folder_path: Union[str, Path, os.PathLike[str]],
    zip_file_path: Union[str, Path, os.PathLike[str]],
    logger: logging.Logger=None,
    ):
    # Extract the contents of the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    msg = f"ZIP file extracted successfully to {folder_path}."
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def organize(folder_path: Union[str, Path, os.PathLike[str]], logger: logging.Logger=None,):
    # Move the contents of the PAMAP2 folder to the downloads folder and delete the PAMAP2 folder
    pamap2_folder_path = os.path.join(folder_path, 'PAMAP2data')
    if os.path.exists(pamap2_folder_path):
        for item in os.listdir(pamap2_folder_path):
            s = os.path.join(pamap2_folder_path, item)
            d = os.path.join(folder_path, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.move(s, d)
        shutil.rmtree(pamap2_folder_path)
        msg = f"PAMAP2 folder contents moved to {folder_path} and PAMAP2 folder deleted."
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

class DownloadError(Exception):
    """Custom exception for download errors."""
    pass

###########
# logging #
###########

def log_stats(
    method: str,
    cli_logger: logging.Logger,
    train_dataset: Dataset, # PamSubDataset,
    val_dataset: Dataset,
    test_dataset: Dataset
    ):
    if method == TrainingMethodOptions.centralized:
        log_stats_centralized(cli_logger=cli_logger, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
    )
    #elif method == TrainingMethodOptions.federated:
        #log_stats_federated()
    else:
        RuntimeError

def log_stats_centralized(
    cli_logger: logging.Logger, 
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    ):
    cli_logger.info("-" * 50)
    cli_logger.info("Statistics")
    cli_logger.info("-" * 50)
    
    cli_logger.info(f"Size of Train Data:\t{len(train_dataset)}")
    cli_logger.info(f"Size of Train Data:\t {len(val_dataset)}")
    cli_logger.info(f"Size of Test Data:\t {len(test_dataset)}")
    cli_logger.info("-" * 50)
    cli_logger.info(f"Total Data Size:\t{len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    cli_logger.info("-" * 50)

    cli_logger.info(f"\n")
    cli_logger.info(f"Label Distribution")
    v, c = np.unique(train_dataset.targets, return_counts=True)
    cli_logger.info("\t\t|" + "  |".join([f' {vv}' for vv in v]) + ' |')
    cli_logger.info(f"Train Dataset\t|" + " |".join([f" {(((vv / len(train_dataset.targets))*100)):.2f}" for vv in c]) + ' |')
    
    v, c = np.unique(val_dataset.targets, return_counts=True)
    cli_logger.info(f"Val Dataset\t|" + " |".join([f" {(((vv / len(val_dataset.targets))*100)):.2f}" for vv in c]) + ' |')
    
    v, c = np.unique(test_dataset.targets, return_counts=True)
    cli_logger.info(f"Test Dataset\t|" + " |".join([f" {(((vv / len(test_dataset.targets))*100)):.2f}" for vv in c]) + ' |')
    cli_logger.info("-" * 50)


    