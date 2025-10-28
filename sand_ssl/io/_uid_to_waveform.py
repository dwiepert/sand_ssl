"""
Load waveform from a uid

Author(s): Daniela Wiepert
Last modified: 10/2025
"""
#IMPORTS
##built-in
from typing import Tuple, Union
from pathlib import Path

##third party
import librosa
import torch
import torchaudio

class UidToWaveform(object):
    '''
    Take a UID, find & load the data, add waveform and sample rate to sample
    :param prefix:str, path prefix for searching
    :param extension:str, audio file extension (default = None)
    :param lib: bool, indicate whether to load with librosa (default=False, uses torchaudio)
    :param structured: bool, indicate whether audio files are in structured format (prefix/uid/waveform.wav) or not (default=False)
    '''
    
    def __init__(self, prefix:Union[str,Path], extension:str='wav', lib:bool=False):
    
        self.prefix = prefix #either gcs_prefix or input_dir prefix
        if not isinstance(self.prefix, Path): self.prefix = Path(self.prefix)
        self.cache = {}
        self.extension = extension
        self.lib = lib
    
    def _load_waveform(self, uid, task):
        '''
        TODO
        '''
        waveform_path = self.prefix / task 
        waveform_path = waveform_path / f'{uid}.{self.extension}'

        if not self.lib:
            waveform, sr = torchaudio.load(waveform_path, format = self.extension)
        else:
            waveform, sr = librosa.load(waveform_path, mono=False, sr=None)
            waveform = torch.from_numpy(waveform)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.shape[1] == 1 or waveform.shape[1] == 2:
                waveform = torch.transpose(waveform)

        return waveform, sr

    def __call__(self, sample:dict) -> dict:
        """
        Load waveform
        :param sample: dict, input sample
        :return wavsample: dict, sample after loading
        """
        wavsample = sample.copy()
        uid, task = wavsample['uid'], wavsample['task']
        cache = {}
        if uid not in self.cache:
            wav, sr = self._load_waveform(uid, task)
            cache['waveform'] = wav 
            cache['sample_rate'] = sr
            self.cache[uid] = cache
            
        cache = self.cache[uid]
        
        wavsample['waveform'] = cache['waveform']
        wavsample['sample_rate'] = cache['sample_rate']
         
        return wavsample
    
