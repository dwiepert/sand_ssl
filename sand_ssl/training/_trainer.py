"""
Custom model trainer

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##built-in
import json
import gc
from pathlib import Path
import os
from typing import Union, List, Tuple

##third party
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, SequentialLR, LambdaLR, CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm

##local
from sand_ssl.models import Model
from sand_ssl.io import save_checkpoint
from ._early_stop import EarlyStopping

#HELPER FOR SCHEDULER
def warmup_lr_lambda(epoch:int, warmup_epochs:int = 5) -> float:
    """
    Function for warmup scheduler that defines when to stop warmup
    :param epoch: int, current epoch
    :param warmup_epochs: int, number of warmup epochs (default = 5)
    :return: fraction of warmup epochs completed
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def warmup_wrapper(warmup_epochs:int = 5):
    """
    Wraper for warmup to flexibly set the number of warmup epochs
    
    :param warmup_epochs: int, number of warmup epochs (default = 5)
    :return: lambda function
    """
    return lambda w: warmup_lr_lambda(w, warmup_epochs=warmup_epochs)

#MAIN CLASS
class Trainer():
    """
    Custom model training class

    :param model: initialized model
    :param target_features: list of target features
    :param optim_type: str, optimizer type (default = adamw)
    :param learning_rate: float, learning rate (default = 1e-3)
    :param loss_type: str, loss type (default = bce)
    :param gradient_accumulation_steps: int, number of steps for gradient accumulation (default = 4)
    :param batch_size: int, batch size (default = 2)
    :param scheduler_type: str, scheduler type (default = None)
    :param early_stop: bool, specify whether to use early stopping (default = False)
    :param save_checkpoints: bool, specify whether to save checkpoints (default = True)
    :param patience: int, patience for early stopping (default = 5)
    :param delta: float, minimum change for early stopping (default = 0.0)
    :param rating_threshold: float, for converting rank to binary
    :param kwargs: additional values for rank classification loss or schedulers (e.g., rating_threshold/margin/bce_weight for rank loss and end_lr/epochs for Exponential scheduler)
    """
    def __init__(self, model:Union[Model], optim_type:str="adamw", learning_rate:float=1e-4, loss_type:str="ce", batch_size:int=2,
                 scheduler_type:str=None, early_stop:bool=False, save_checkpoints:bool=True, patience:int=5, delta:float=0.0,**kwargs):
        
        self.model = model
        self.optim_type = optim_type
        self.learning_rate= learning_rate
        self.batch_size = batch_size
        self.best_model = None
        
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.delta = delta
        self.config = {'learning_rate': self.learning_rate, 'optim_type':self.optim_type, 'loss_type': self.loss_type, 'scheduler_type':self.scheduler_type,
                        'batch_size':self.batch_size}

        self.config.update(self.model.config)

        self.save_checkpoints = save_checkpoints
        
        self._set_up_optimizer()
        self._set_up_loss()
        self._set_up_scheduler(kwargs)

        self._set_up_early_stop(early_stop, kwargs)

        self._generate_save_name()

        self.log = {"train_loss":[], "avg_train_loss":[], "val_loss":[], "avg_val_loss":[]}
        self.log.update(self.config)

        self.model_fit = False

    def _set_up_optimizer(self):
        """
        """
        if self.optim_type == 'adamw':
             self.optim = torch.optim.AdamW(params=self.model.classifier_head.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f'{self.optim_type} not implemented.')
        
    def _set_up_loss(self):
        """
        """
        if self.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f'{self.loss_type} not implemented.')
    
    def _set_up_scheduler(self, kwargs):
        """
        """
        if self.scheduler_type == 'exponential':
            if 'gamma' in kwargs:
                clf_gamma = kwargs['gamma']
                self.clf_gamma = clf_gamma

                self.config['gamma'] = self.clf_gamma
            else:
                self.clf_gamma = None
                assert 'end_lr' in kwargs, 'Must give end_lr if using exponential learning rate decay scheduler.'
                assert 'epochs' in kwargs, 'Must give epochs if using exponential learning rate decay scheduler.'
                end_clf_lr = kwargs['end_lr']
                epochs = kwargs['epochs']
                clf_gamma = end_clf_lr / (self.learning_rate**(1/epochs))
                self.end_clf_lr = end_clf_lr
                self.epochs = epochs
                self.config['end_lr'] = self.end_clf_lr
                self.config['epochs'] = self.epochs
            
            self.scheduler = ExponentialLR(self.clf_optim, gamma=clf_gamma)

        elif self.scheduler_type == 'warmup-cosine':
            assert 'warmup_epochs' in kwargs,'Must give warmup_epochs if using warmup.'
            assert 'train_len' in kwargs, 'Must give train_len if using cosine annealing lr.'
            clf_warmup_e = kwargs['warmup_epochs']
            
            self.clf_warmup_e = clf_warmup_e 
            self.train_len = kwargs['train_len']
            self.config['warmup_epochs'] = self.clf_warmup_e
            self.config['train_len'] =self.train_len

            clf_warmup = LambdaLR(self.clf_optim, lr_lambda=warmup_wrapper(clf_warmup_e))
            clf_cosine = CosineAnnealingLR(self.clf_optim, T_max = self.train_len)
            self.scheduler = SequentialLR(self.clf_optim, schedulers=[clf_warmup,clf_cosine], milestones=[clf_warmup_e-1]) 
        
        elif self.scheduler_type == 'cosine':
            assert 'train_len' in kwargs, 'Must give train_len if using cosine annealing lr.'
            self.train_len = kwargs['train_len']
            self.config['train_len'] = self.train_len
            self.scheduler = CosineAnnealingLR(self.clf_optim, T_max = kwargs['train_len'])
        elif self.scheduler_type is not None:
            raise NotImplementedError(f'{self.scheduler_type} not implemented.')
        else:
            self.scheduler = None

    def _set_up_early_stop(self, early_stop, kwargs):
        """
        """
        #es 
        if early_stop:
            self.config['early_stop'] = True
            es_params = {'patience':self.patience, 'delta':self.delta}
            self.config.update(es_params)
            if 'test' in kwargs:
                es_params['test'] = kwargs.pop('test')
            self.early_stop = EarlyStopping(**es_params)
        else:
            self.early_stop = None

    def _generate_save_name(self):
        """
        """
        print('pause')
        #MODEL COMPONENT
        self.name_prefix = self.model.get_model_name()
        
        
        #TRAINING COMPONENT
        self.name_prefix += f'_{self.optim_type}_lr{self.learning_rate}_{self.loss_type}'
        
    
        if self.scheduler_type == 'exponential':
            if self.clf_gamma:
                self.name_prefix += f'_{self.scheduler_type}_g{self.clf_gamma}'
            else:
                self.name_prefix += f'_{self.scheduler_type}_endlr{self.end_clf_lr}'
        elif self.scheduler_type == 'warmup-cosine':
            self.name_prefix += f'_{self.scheduler_type}_we{self.clf_warmup_e}_tl{self.train_len}'
        elif self.scheduler_type == 'cosine':
            self.name_prefix += f'_{self.scheduler_type}_tl{self.train_len}'

        if self.early_stop:
            self.name_prefix += '_es'

        self.name_prefix += f'_bs{self.batch_size}'
  
    def train_step(self, train_loader:DataLoader):
        """
        Training step
        :param train_loader: DataLoader with training data
        """
        self.model.train()
        running_loss = 0.

        for index, data in enumerate(tqdm(train_loader)):
            

            inputs, attn_mask, target = data['waveform'].to(self.model.device), data['attn_mask'].to(self.model.device), data['target'].to(self.model.device)
            
            #self.tf_optim.zero_grad()
            #self.clf_optim.zero_grad()

            outputs = self.model(inputs, attn_mask)
            #prediction = torch.argmax(outputs, dim=1).type(torch.float32).to(self.model.device)
            
            self.optim.zero_grad()
            loss = self.criterion(outputs, target)
            loss.backward()
            running_loss += loss.item()
    

            self.optim.step()
            
            
        self.log['train_loss'].append(running_loss)
        self.log['avg_train_loss'].append((running_loss / len(train_loader)))
        
        #self.fit = True

    def val_step(self, val_loader:DataLoader, e:int):
        """
        Validation step
        :param val_loader: Dataloader with validation data
        :param e: int, current epoch
        """
        self.model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader):
                
                inputs, attn_mask, target = data['waveform'].to(self.model.device), data['attn_mask'].to(self.model.device), data['target'].to(self.model.device)
    
                outputs = self.model(inputs, attn_mask)

                loss = self.criterion(outputs, target)
                running_vloss += loss.item()



        self.log['val_loss'].append(running_vloss)
        self.log['avg_val_loss'].append((running_vloss / len(val_loader)))

        if self.early_stop:
            self.early_stop(running_vloss, self.model, e)
        
    def fit(self, train_loader:DataLoader, out_dir:Union[Path,str], val_loader:DataLoader=None, epochs:int=1):
        """
        Fit the model with train_loader and optional val_loader

        :param tain_loader: DataLoader with training data
        :param val_loader: DataLoader with validation data (default=None)
        :param epochs: int, number of epochs to train for
        :param: TODO
        """
        if not isinstance(out_dir, Path): out_dir = Path(out_dir)

        #FLUSH CONFIG
        self.epochs = epochs
        self.config['epochs'] = self.epochs

        name_prefix =  f'{self.name_prefix}_e{epochs}'
        
        self.path = out_dir / name_prefix
       # if self.path.exists():
        #    raise ValueError(f'Trying to overwrite a file. Please check values: {self.path}.')
        self.path.mkdir(parents = True, exist_ok = True)

        config_path = self.path / 'config.json'
        with open(str(config_path), 'w') as f:
            json.dump(self.config, f, indent=4)
 
        for e in range(epochs):
            self.train_step(train_loader)

            if val_loader:
                self.val_step(val_loader, e)

            log_path = self.path / 'train_log.json'
            with open(str(log_path), 'w') as f:
                json.dump(self.log, f, indent=4)

            
            if self.early_stop:
                if self.early_stop.early_stop:
                    best_model, best_epoch, _ = self.early_stop.get_best_model()
                    out_name = Path(name_prefix) /f'best{best_epoch}'
                    save_checkpoint(best_model, out_dir, sub_dir=out_name)
                    self.best_model = best_model
                    self.best_epoch = best_epoch
                    self.model_fit = True
                    break
            
            #checkpointing
            if (e ==0 or e % 5 == 0) and (e != epochs - 1) and self.save_checkpoints:
                out_name = Path(name_prefix) /f'checkpoint{e}'
                #TODO
                save_checkpoint(self.model, out_dir, sub_dir=out_name)
            
            if self.scheduler:
                self.scheduler.step()
        
            if e == epochs - 1:
                out_name = Path(name_prefix) /f'final{e}'
                save_checkpoint(self.model, out_dir, sub_dir=out_name)

                if self.early_stop:
                    best_model, best_epoch, _ = self.early_stop.get_best_model()
                    out_name = Path(name_prefix) /f'best{best_epoch}'
                    save_checkpoint(best_model, out_dir, sub_dir=out_name)
                    self.best_model = best_model
                    self.best_epoch = best_epoch

            self.model_fit = True 

    def test(self, test_loader:DataLoader, out_dir:Union[Path,str], test_best:bool=True):
        """
        Evaluate model on test data

        :param testloader: DataLoader with test data
        """
        if not isinstance(out_dir, Path): out_dir = Path(out_dir)

        if self.best_model is not None and test_best:
            model = self.best_model
        else:
            model = self.model 

        if not self.model_fit:
            name_prefix = self.name_prefix
            self.path = out_dir / name_prefix
            if self.path.exists():
                raise ValueError(f'Trying to overwrite a file. Please check values: {self.path}.')
            self.path.mkdir(parents = True, exist_ok = True)
            config_path = self.path / 'config.json'
            with open(str(config_path), 'w') as f:
                json.dump(self.config, f, indent=4)

            
        model.eval()
        running_loss = 0.0
        
        output_list = []
        target_list = []
        prediction_list = []
        metrics = {}

        sm = nn.Softmax()
        with torch.no_grad():
            running_loss = 0.0
            for data in tqdm(test_loader):
                inputs, attn_mask, target = data['waveform'].to(model.device), data['attn_mask'].to(model.device), data['target'].to(model.device)
                
                outputs = model(inputs, attn_mask)
                prediction = torch.argmax(sm(outputs), dim=1)
                
                loss = self.criterion(outputs, target)
                running_loss += loss.item()

                outputs = outputs.cpu()
                target = target.cpu()
                prediction = prediction.cpu()
                
                output_list.append(outputs)
                prediction_list.append(prediction)
                target_list.append(target)      
        
        outputs = torch.vstack(output_list)
        targets = torch.concat(target_list)
        predictions= torch.concat(prediction_list)
        
        f1 = f1_score(targets, predictions, average='macro')

        metrics['f1'] = f1
        metrics['test_loss'] =  running_loss
        metrics['avg_test_loss'] = (running_loss / len(test_loader))

        if self.best_model:
            metrics['best_epoch'] = self.best_epoch

        log_path = self.path / 'evaluation.json'
        
        with open(str(self.path / 'evaluation.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        

    def _binarize(self, outputs:torch.Tensor, targets:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate binary evaluation metrics

        :params outputs: torch.Tensor, model outputs
        :params targets: torch.Tensor, target labels (n_test, n_features)
        :return output_probabilities: torch.Tensor, model probabilities
        :params binary_targets: torch.Tensor, binary target labels (n_test, n_features)
        """

        #binary targets: 
        binary_targets = (targets >= self.rating_threshold).float()
        activation = torch.nn.Sigmoid()
        output_probabilities = activation(outputs) 
        #binary_outputs = (output_probabilities > 0.5).float()
        return output_probabilities, binary_targets #binary_outputs, binary_targets
    
    def _binary_metrics(self, probabilities:np.ndarray, targets:np.ndarray) -> dict:
        """
        TODO
        """
        binary_outputs= (probabilities > 0.5).astype('float')
        #OVERALL AUC
        macro_auc = roc_auc_score(targets,probabilities, average='macro')
        weighted_auc = roc_auc_score(targets,probabilities, average='weighted')
        sample_auc = roc_auc_score(targets,probabilities, average='samples')

        #OVERALL ACCURACY
        overall_acc = accuracy_score(targets, binary_outputs)

        bacc_per_feature = []
        acc_per_feature = []
        auc_per_feature = []
        for i in range(len(self.target_features)):
            bacc_per_feature.append(balanced_accuracy_score(targets[:,i], binary_outputs[:,i]))
            acc_per_feature.append(accuracy_score(targets[:,i], binary_outputs[:,i]))
            auc_per_feature.append(roc_auc_score(targets[:,i], probabilities[:,i]))
  
        return {'probabilities':probabilities.tolist(), 'binary_targets':targets.tolist(), 'macro_auc': macro_auc, 'weighted_auc':weighted_auc, 'sample_auc':sample_auc,
                'overall_acc':overall_acc, 'bacc_per_feature':bacc_per_feature, 'acc_per_feature':acc_per_feature, 'auc_per_feature': auc_per_feature, 'target_features':self.target_features}
        
    def _rating_metrics(self, outputs:np.ndarray, targets:np.ndarray) -> dict:
        """
        TODO
        """    
        #overral correlation
        rho, pval = spearmanr(targets, outputs, axis=None)

        #per feature
        rho_per_feature = []
        p_per_feature = []
        for i in range(len(self.target_features)):
            r, p = spearmanr(outputs[:,i], targets[:,i])
            rho_per_feature.append(r.item())
            p_per_feature.append(p.item())
        
        return {'outputs': outputs.tolist(), 'targets':targets.tolist(), 'overall_rho':rho.item(), 'overall_p':pval.item(), 'target_features':self.target_features, 'rho_per_feature':rho_per_feature, 'p_per_feature':p_per_feature}


