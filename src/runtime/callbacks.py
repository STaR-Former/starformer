import matplotlib.pyplot as plt
import wandb
import numpy as np
from typing import Any, Mapping, List, Union
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import fbeta_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only

import torch 
from torch import Tensor

from src.util import DatasetOptions, ModelOptions
from omegaconf import DictConfig

__all__ = [
    'LossLoggingCallback',
    'MetricLoggingCallback'
]

"""
refactor
"""

class LossLoggingCallback(Callback):
    def __init__(self, config: DictConfig,) -> None:
        super().__init__()
        self.config = config

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_wand(pl_module)
        return super().on_fit_start(trainer, pl_module)

    #@rank_zero_only
    def setup_wand(self, pl_module: LightningModule):
       if self.config.logger.name == 'wandb':
            #wandb.define_metric("train/loss", summary="min")
            #wandb.define_metric("val/loss", summary="min") 
            pl_module.logger.experiment.define_metric("train/loss", summary="min")
            pl_module.logger.experiment.define_metric("val/loss", summary="min")

    #def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        # log loss
        self.__log_loss(trainer, pl_module, outputs=outputs, mode="train")

        #return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    #def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the validation batch ends."""
        # log loss
        self.__log_loss(trainer, pl_module, outputs=outputs, mode="val")
    
    #def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the test batch ends."""
        # log loss
        self.__log_loss(trainer, pl_module, outputs=outputs, mode="test")
    
    #def __log_loss(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, mode: str="train") -> None:
    def __log_loss(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], mode: str="train") -> None:
        loss = outputs['loss']
        if self.config.model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'drm_loss':
            if self.config.model.masking is not None:
                if mode == 'test':
                     self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                elif mode == 'val': # or mode == 'test':
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_mse': outputs['loss_mse']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_mse_masked': outputs['loss_mse_masked']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_mse_unmasked': outputs['loss_mse_unmasked']}, batch_size=outputs['bs'], sync_dist=True)
                else:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                    self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_mse': outputs['loss_mse']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_mse_masked': outputs['loss_mse_masked']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_mse_unmasked': outputs['loss_mse_unmasked']}, batch_size=outputs['bs'], sync_dist=False)
            else:
                if mode in ['val', 'test']:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)    
                else:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)
        
        elif self.config.model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'contrastive_loss':
            if self.config.model.masking is not None:
                if mode == 'test':
                     self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                elif mode == 'val': # or mode == 'test':
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=True)
                    self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=True)
                    #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=True)
                else:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                    self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=False)
                    self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=False)
                    #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=False)
            else:
                if mode in ['val', 'test']:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)    
                else:
                    self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)                
        else:           
            if mode in ['val', 'test']:
                self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)    
            else:
                self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)
    

class MetricLoggingCallback(Callback):
    def __init__(self, config: DictConfig, log_cm_train: bool=False, log_cm_val: bool=False) -> None:
        super().__init__()
        self.validation_step_outputs = []
        self.testing_step_outputs = []
        self.config = config
        self.log_cm_train = log_cm_train
        self.log_cm_val = log_cm_val
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_wandb(pl_module)
        return super().on_fit_start(trainer, pl_module)

    #@rank_zero_only
    def setup_wandb(self, pl_module: LightningModule):
        if self.config.logger.name == 'wandb':
            pl_module.logger.experiment.define_metric("train/acc", summary="max")
            pl_module.logger.experiment.define_metric("train/fbeta", summary="max")
            pl_module.logger.experiment.define_metric("val/acc", summary="max")
            pl_module.logger.experiment.define_metric("val/fbeta", summary="max")
            
            #wandb.define_metric("train/acc", summary="max")
            #wandb.define_metric("train/fbeta", summary="max")
            #wandb.define_metric("val/acc", summary="max")
            #wandb.define_metric("val/fbeta", summary="max")
        #if self.config.dataset == 'geolife':
        #    self.testing_acc_per_label = {
        #        k: {'pred': [], 'truth': []} 
        #        for k in range(len(list(trainer.datamodule.dataset.idx2label.keys())))
        #    }


    #def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        mode = 'train'

        if pl_module.config.dataset in [DatasetOptions.geolife]:
            preds = outputs['preds'].flatten()
            labels = outputs['labels'].flatten()
            bs = outputs['bs']
            # fbeta calc and log
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=outputs['bs'], sync_dist=False, prog_bar=True, logger=True) #, on_step=False, on_epoch=True)
            #trainer.logger.experiment.add_scalar(f'{mode}/fbeta', fbeta, )

            # confusion matrix calc and log
            if self.log_cm_train:
                cm_matrix = outputs['cm']
                #tn, fp, fn, tp = cm_matrix.ravel()
                if self.config.datamodule.get('display_labels', None) is not None:
                    display_labels = self.config.datamodule.display_labels
                else:
                    display_labels = list(trainer.datamodule.dataset.idx2label.values())
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
                
                if self.config.logger.name == 'tensorboard':
                    trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)
                elif self.config.logger.name == 'wandb':
                    #self.__log_img(self, trainer, pl_module, mode, title="Confusion Matrix", img=cm_fig)
                    trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
                
        elif pl_module.config.dataset in DatasetOptions.ucr_uea or pl_module.config.dataset == DatasetOptions.pam:
            preds = outputs['preds'].flatten()
            labels = outputs['labels'].flatten()
            # fbeta calc and log
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=outputs['bs'], sync_dist=False, prog_bar=True, logger=True) #, on_step=False, on_epoch=True)

        # log loss'
        self.__log_acc(trainer, pl_module, outputs=outputs, mode="train")

        #return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    #def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the validation batch ends."""
        self.validation_step_outputs.append(outputs)
        # log acc
        self.__log_acc(trainer, pl_module, outputs=outputs, mode="val")
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mode = 'val'
        if pl_module.config.dataset in [DatasetOptions.geolife]:
            preds = []
            labels = []
            cms = []
            bs = self.validation_step_outputs[0]['bs']

            for batch_outputs in self.validation_step_outputs:
                preds.append(batch_outputs['preds'].flatten())
                labels.append(batch_outputs['labels'].flatten())
                cms.append(batch_outputs['cm'])

            # fbeta calc and log    
            labels = torch.concat(labels)
            preds = torch.concat(preds)
            
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=bs, sync_dist=True)#, on_step=False, on_epoch=True) 
            #trainer.logger.experiment.add_scalar(f'{mode}/fbeta', fbeta, trainer.current_epoch)

            # confusion matrix calc and log
            if self.log_cm_val:
                cm_matrix = sum(cms)
                #tn, fp, fn, tp = cm_matrix.ravel()
                if self.config.datamodule.get('display_labels', None) is not None:
                    display_labels = self.config.datamodule.display_labels
                else:
                    display_labels = list(trainer.datamodule.dataset.idx2label.values())
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
                if self.config.logger.name == 'tensorboard':
                    trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)
                elif self.config.logger.name == 'wandb':
                    #self.__log_img(self, trainer=trainer, pl_module=pl_module, 
                    #   mode=mode, title="Confusion Matrix", img=cm_fig)
                    trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
            
            if pl_module.config.dataset == DatasetOptions.geolife:
                self.__log_geolife_per_class_scores_tabel(trainer=trainer, preds=preds, labels=labels, mode=mode)
        
        elif pl_module.config.dataset in DatasetOptions.ucr_uea or pl_module.config.dataset == DatasetOptions.pam:
            preds, labels, cms = [], [], []
            bs = self.validation_step_outputs[0]['bs']

            for batch_outputs in self.validation_step_outputs:
                preds.append(batch_outputs['preds'].flatten())
                labels.append(batch_outputs['labels'].flatten())
                cms.append(batch_outputs['cm'])

            # fbeta calc and log    
            labels = torch.concat(labels)
            preds = torch.concat(preds)
            
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=bs, sync_dist=True)#, on_step=False, on_epoch=True) 

             # confusion matrix
            if self.config.logger.name == 'wandb':
                cm_matrix = sum(cms)
                #tn, fp, fn, tp = cm_matrix.ravel()
                display_labels = self.config.datamodule.display_labels
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])

        self.validation_step_outputs.clear()
    

    #def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the test batch ends."""
        self.testing_step_outputs.append(outputs)
        batch = outputs['batch']
        # log acc
        self.__log_acc(trainer, pl_module, outputs=outputs, mode="test")
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mode = 'test'
        if pl_module.config.dataset in [DatasetOptions.geolife]:
            preds = []
            labels = []
            cms = []
            bs = self.testing_step_outputs[0]['bs']

            for batch_outputs in self.testing_step_outputs:
                preds.append(batch_outputs['preds'].flatten())
                labels.append(batch_outputs['labels'].flatten())
                cms.append(batch_outputs['cm'])
            
            # fbeta calc and log
            labels = torch.concat(labels)
            preds = torch.concat(preds)
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=bs, sync_dist=True) 
            #trainer.logger.experiment.add_scalar(f'{mode}/fbeta', fbeta, trainer.global_step)
            
            # log res in tabel
            columns = ['acc', 'fbeta']
            acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self.testing_step_outputs]).mean().item()
            self.__log_table(trainer, key="Scores", columns=columns, data=[[acc, fbeta]], mode=mode)

            # confusion matrix calc and log
            cm_matrix = sum(cms)

            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = list(trainer.datamodule.dataset.idx2label.values())
            cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

            if self.config.logger.name == 'tensorboard':
                trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)
            
            elif self.config.logger.name == 'wandb':
                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])

            if pl_module.config.dataset == DatasetOptions.geolife:
                self.__log_geolife_per_class_scores_tabel(trainer=trainer, preds=preds, labels=labels, mode=mode)


        if pl_module.config.dataset in DatasetOptions.ucr_uea or pl_module.config.dataset == DatasetOptions.pam:
            preds = []
            labels = []
            cms = []
            bs = self.testing_step_outputs[0]['bs']

            for batch_outputs in self.testing_step_outputs:
                preds.append(batch_outputs['preds'].flatten())
                labels.append(batch_outputs['labels'].flatten())
                cms.append(batch_outputs['cm'])
            
            # fbeta calc and log
            labels = torch.concat(labels)
            preds = torch.concat(preds)
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #trainer.logger.experiment.add_scalar(f'{mode}/fbeta', fbeta, trainer.global_step)
            
            # log res in tabel
            columns = ['acc', 'fbeta']
            acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self.testing_step_outputs]).mean().item()
            self.__log_table(trainer, key="Scores", columns=columns, data=[[acc, fbeta]], mode=mode)

            # confusion matrix
            if self.config.logger.name == 'wandb':
                cm_matrix = sum(cms)
                #tn, fp, fn, tp = cm_matrix.ravel()
                display_labels = self.config.datamodule.display_labels
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
                # self.__log_img(self, trainer, pl_module, mode, title="Confusion Matrix", img=cm_fig)
        
        if pl_module.config.dataset == DatasetOptions.pam:
            preds = []
            labels = []
            cms = []
            bs = self.testing_step_outputs[0]['bs']

            for batch_outputs in self.testing_step_outputs:
                preds.append(batch_outputs['preds'].flatten())
                labels.append(batch_outputs['labels'].flatten())
                cms.append(batch_outputs['cm'])
            
            # fbeta calc and log
            labels = torch.concat(labels)
            preds = torch.concat(preds)
            fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
            self.__log_fbeta(fbeta, bs=bs, mode=mode)
            #trainer.logger.experiment.add_scalar(f'{mode}/fbeta', fbeta, trainer.global_step)
            
            # log res in tabel
            columns = ['acc', 'fbeta']
            acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self.testing_step_outputs]).mean().item()
            self.__log_table(trainer, key="Scores", columns=columns, data=[[acc, fbeta]], mode=mode)

            # confusion matrix
            if self.config.logger.name == 'wandb':
                cm_matrix = sum(cms)
                #tn, fp, fn, tp = cm_matrix.ravel()
                display_labels = self.config.datamodule.display_labels
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
                # self.__log_img(self, trainer, pl_module, mode, title="Confusion Matrix", img=cm_fig)

            # Calculate macro-averaged precision, recall, and F1 score
            precision_macro = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
            recall_macro = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
            f1_macro = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
            self.log_dict({f'{mode}/precision': precision_macro, f'{mode}/recall': recall_macro, f'{mode}/f1_score': f1_macro}, batch_size=bs, sync_dist=True, prog_bar=True, logger=True) 
            
        self.testing_step_outputs.clear()
    
    #def __log_acc(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, mode: str="train") -> None:
    #@rank_zero_only
    def __log_img(self, trainer: Trainer, mode: str="train", title: str=None, img=None) -> None:
        assert img is not None
        trainer.logger.log_image(key=f'{mode}/{title}', images=[img])

    #@rank_zero_only
    def __log_fbeta(self, fbeta: float=None, bs: int=None, mode: str="train") -> None:
        assert fbeta is not None
        assert bs is not None
        if mode in ['val', 'test']:
            self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=bs, sync_dist=True, prog_bar=True, logger=True) 
        else:
            self.log_dict({f'{mode}/fbeta': fbeta}, batch_size=bs, sync_dist=False, prog_bar=True, logger=True) 

    #@rank_zero_only
    def __log_acc(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], mode: str="train") -> None:
        if outputs.get('acc', None) is not None:
            acc = outputs['acc'].item() if isinstance(outputs['acc'], Tensor) else outputs['acc']
            if mode in ['val', 'test']:
                self.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=True, prog_bar=True, logger=True)    
                #self.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, logger=True)    
            else:
                #pl_module.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=False, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=False, prog_bar=True, logger=True)
        
        if outputs.get('acc_autoregressive', None) is not None:
            acc = outputs['acc_autoregressive'].item() if isinstance(outputs['acc_autoregressive'], Tensor) else outputs['acc_autoregressive']
            if mode in ['val', 'test']:
                self.log_dict({f'{mode}/acc_autoregressive': acc}, batch_size=outputs['bs'], sync_dist=True, prog_bar=True, logger=True)    
                #self.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, logger=True)    
            else:
                #pl_module.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=False, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log_dict({f'{mode}/acc_autoregressive': acc}, batch_size=outputs['bs'], sync_dist=False, prog_bar=True, logger=True)
    
    def __create_cm_figure(self, cm_matrix, display_labels: List=[0,1], cmap: str='viridis'):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=display_labels)
        disp.plot(cmap=cmap)

        plt.close(disp.figure_)
        return disp.figure_

    
    #@rank_zero_only
    def __log_geolife_per_class_scores_tabel(self, trainer: Trainer, preds: Tensor, labels: Tensor, mode: str='val'):
        fbeta_per_class = fbeta_score(
            labels.detach().cpu().numpy(), 
            preds.detach().cpu().numpy(), 
            average=None, beta=0.5)

        sorted_by_label = {
            k: {'pred': [], 'truth': []} 
            for k in range(len(list(trainer.datamodule.dataset.idx2label.keys())))
        }
        for p, l in zip(preds, labels):
            sorted_by_label[l.item()]['pred'].append(p.item())
            sorted_by_label[l.item()]['truth'].append(l.item())
        
        acc_per_label = {}
        for k in sorted_by_label.keys():
            if acc_per_label.get(k, None) is None:
                acc_per_label[k] = {'acc': None, 'fbeta': None}
            
            #print(sorted_by_label[k]['truth'], bool(sorted_by_label[k]['truth']))
            if bool(sorted_by_label[k]['truth']):
                acc_per_label[k]['acc'] = (torch.sum(
                            torch.eq(torch.tensor(sorted_by_label[k]['pred']), 
                            torch.tensor(sorted_by_label[k]['truth']))
                        ) / len(torch.tensor(sorted_by_label[k]['truth']))).item()
                
                acc_per_label[k]['fbeta'] = fbeta_per_class[k] if k <= len(fbeta_per_class)-1 else 0.0 
            else:
                acc_per_label[k]['acc'] = 0.0
                acc_per_label[k]['fbeta'] = 0.0
        

        # accuracy
        #columns = ['score'] + [trainer.datamodule.dataset.idx2label[k] for k in acc_per_label.keys()]
        #acc_data = ['acc'] + [acc_per_label[k]['acc'] for k in acc_per_label.keys()]
        #fbeta_data = ['fbeta'] + [acc_per_label[k]['fbeta'] for k in acc_per_label.keys()]
        #data = [acc_data, fbeta_data]
        columns = [trainer.datamodule.dataset.idx2label[k] for k in acc_per_label.keys()]
        acc_data = [acc_per_label[k]['acc'] for k in acc_per_label.keys()]
        fbeta_data = [acc_per_label[k]['fbeta'] for k in acc_per_label.keys()]
        self.__log_table(trainer, key="Accuracy", columns=columns, data=[acc_data], mode=mode)
        self.__log_table(trainer, key="Fbeta", columns=columns, data=[fbeta_data], mode=mode)
    
    def __log_table(self, trainer: Trainer, key: str, columns: list, data: list, mode: str='val'):
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table(key=f'{mode}/{key}', columns=columns, data=data)
