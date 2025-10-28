"""
Base dataset class

Author(s): Daniela Wiepert
Last modified: 10/2025
"""
#IMPORTS
##built-in
from typing import List, Union 
from pathlib import Path

##third-party
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np

##local
from ._uid_to_waveform import UidToWaveform
from ._resample import ResampleAudio
from ._to_monophonic import ToMonophonic
from ._truncate import Truncate
from ._one_hot import OneHotEncode
from ._load_spreadsheet import _load_spreadsheet

class CustomDataset(Dataset):
    '''
    Simple audio dataset
    '''
    
    def __init__(self, audio_dir:Union[Path, str], spreadsheet_path:Union[str,Path], metadata_path:Union[str,Path]=None, data_type:str='training', id_col:str='ID', 
                    target_label:str='Class', tasks:Union[List[str]]=['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU', 'rhythmKA', 'rhythmPA', 'rhythmTA'],
                    use_librosa:bool=False, resample_rate:int=16000, truncate:int=None, debug:bool=False):
        '''
        Initialize dataset with dataframe, target labels, and list of transforms
        :param data: pd.DataFrame, table with all annotation values
        :param uid_col: str, specify which column is the uid col
        :param target_labels: List[str], list of target columns in data
        :param transforms: torchvision transforms function to run on data (default=None)
        '''
        super(CustomDataset, self).__init__()

        self.audio_dir = audio_dir
        if not isinstance(self.audio_dir, Path): self.audio_dir = Path(self.audio_dir)

        self.spreadsheet_path = spreadsheet_path
        if not isinstance(self.spreadsheet_path, Path): self.spreadsheet_path = Path(spreadsheet_path)

        self.metadata_path = metadata_path 
        if self.metadata_path:
            if not isinstance(self.metadata_path, Path): self.metadata_path = Path(metadata_path)

        self.data_type = data_type

        self.id_col = id_col
        self.target_label = target_label
        self.tasks = tasks 
        self.use_librosa = use_librosa
        self.resample_rate = resample_rate
        self.truncate = truncate
        self.task_col = 'task'
        self.debug = debug

        self.data = _load_spreadsheet(self.spreadsheet_path, self.metadata_path, self.data_type, self.tasks, id_col)
        self.uid_col = 'uid'
        
        assert isinstance(self.data, pd.DataFrame), 'Must give dataframe'
        if self.data.index.name != self.uid_col:
            assert self.uid_col in self.data, 'UID column must be present in dataset.'
            self.data = self.data.set_index(self.uid_col)
            
        if self.debug:
            self.data= self.data.sample(n=100)
        assert self.target_label is not None, 'Must give target label.'
        assert self.target_label in self.data.columns.to_list()

        self.transforms_list = [UidToWaveform(prefix=self.audio_dir, lib=self.use_librosa), 
                                ResampleAudio(resample_rate = self.resample_rate),
                                ToMonophonic(),
                                OneHotEncode(num_classes=(int(np.max(self.data[self.target_label]))-int(np.min(self.data[self.target_label]))+1))]
        if self.truncate:
            self.transforms_list.append(Truncate(length=self.truncate))
        
        self.transforms = torchvision.transforms.Compose(self.transforms_list)

    def get_data(self, only_targets:bool=False) -> pd.DataFrame:
        '''
        Return dataframe
        :param only_targets:bool, specify whether to only get target label columns from data
        :return self.data: pd.DataFrame
        '''
        if only_targets:
            return self.data[self.target_label]
        
        return self.data
   
    def __len__(self) -> int:
        '''
        Get dataset size
        :return: int, len of dataset 
        '''
        return len(self.data)
    
    def __getitem__(self, idx:Union[int, torch.Tensor, List[int]]) -> dict:
        '''
        Run transformation
        :param idx: index as int
        :return: dict, transformed data sample, Expects that the uid column has been set to index
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list): assert len(idx) == 1, 'Should only have one idx even if given as a tensor.'
        
        uid = self.data.index[idx]

        target = int(self.data[self.target_label].iloc[idx])
        
        task = self.data[self.task_col].iloc[idx]

        sample = {
            'uid' : uid,
            'target' : target,
            'task': task
        }

        return self.transforms(sample)