"""
Convert labels to one hot torch tensor - VERY SPECIFIC

Author(s): Dani Wiepert
Last modified: 10/2025
"""
#IMPORTS
##third-party
import torch
import torch.nn.functional as F

class OneHotEncode(object):
    '''
    Convert labels to a tensor rather than ndarray
    '''
    def __init__(self, num_classes:int=5):
        self.num_classes = num_classes 

    def __call__(self, sample:dict) -> dict:
        """
        :param sample: dict, input sample
        :return tensample: dict, sample after torch conversion
        """
        
        tensample = sample.copy()
        tensample['target_onehot'] = F.one_hot(torch.tensor(tensample['target']-1), num_classes=self.num_classes).type(torch.float32)
        tensample['target'] = torch.tensor(tensample['target']-1).type(torch.long)
        return tensample