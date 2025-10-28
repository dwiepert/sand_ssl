"""
Early stopping

Source: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
Last modified: 07/2025
"""
# IMPORTS
## built-in
from typing import Union, Tuple
## local
from sand_ssl.models import Model

class EarlyStopping:
    """
    Early stopping class

    :param patience: Number of epochs to wait before stopping if no improvement. (default = 5)
    :param delta: float, Minimum change in the monitored quantity to qualify as an improvement. (default = 0)
    :param test: bool, DEBUGGING ONLY (default=False)
    """
    def __init__(self, patience:int=5, delta:float=0.0, test:bool=False):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_model = None
        self.early_stop = False 
        self.test = test

    def __call__(self, val_loss:float, model:Union[Model], epoch: int):
        """
        Keep track of best loss and best model

        :param val_loss: float, validation loss
        :param model: current model
        :param epoch: int, current epoch
        """
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            self.best_epoch = epoch
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.best_epoch = epoch
            self.counter = 0
        
        if self.test:
            self.early_stop=True
    
    def get_best_model(self) -> Tuple[Union[Model],int,float]:
        """
        :return self.best_model: best model during training
        :return self.best_epoch: best epoch
        :return self.best_score: best loss value
        """
        return self.best_model, self.best_epoch, self.best_score