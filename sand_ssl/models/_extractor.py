"""
Hugging Face feature extractor

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORT
##built-in
import os
from pathlib import Path
import shutil
from typing import Union, Optional, Tuple

##third-party
import torch
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor

##local
from sand_ssl.constants import _MODELS

class Extractor():
    """
    Base extractor

    :param model_type: str, type of model being initialized
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt
    :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
    :param normalize: bool, specify whether to normalize audio
    :param bucket: gcs bucket
    :param test_hub_fail: bool, TESTING ONLY
    :param test_local_fail: bool, TESTING ONLY
    """
    def __init__(self, model_type:str, normalize:bool=False):
        
        super(Extractor, self).__init__()
        self.model_type = model_type
        self.normalize = normalize 

        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'

        if _MODELS[self.model_type]['use_featext']:
            self.hf_hub = _MODELS[self.model_type]['hf_hub']
            self._load_extractor()
            self._set_kwargs()

        else:
            self.feature_extractor=None
    
    ### private helpers ###
    def _load_extractor(self):
        """
        Load hugging face extractor
        
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_hub, trust_remote_code=True)
        
    def _set_kwargs(self):
        """
        Set kwargs for feature extractor
        """
        self.feature_extractor_kwargs = {}
        self.feature_extractor_kwargs['return_attention_mask'] = True
        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            self.features_key = 'input_features'
            self.attention_key = 'attention_mask'
        else:
            self.features_key = 'input_values'
            self.attention_key = 'attention_mask'
            self.feature_extractor_kwargs['do_normalize'] = self.normalize
            self.feature_extractor_kwargs['padding'] = True
    
    ### main function(s) ###
    def __call__(self, sample:dict) -> dict:
        """
        Run feature extraction on a sample
        :param sample: dict
        :return new_sample: dict
        """
        new_sample = sample.copy()
        wav = new_sample['waveform']

        if self.feature_extractor:
            wav = [torch.squeeze(w).numpy() for w in wav]
            preprocessed_wav = self.feature_extractor(wav,
                                                        return_tensors='pt', 
                                                        sampling_rate = self.feature_extractor.sampling_rate,
                                                        **self.feature_extractor_kwargs)
            new_sample['waveform'] = preprocessed_wav[self.features_key]
            new_sample['attn_mask'] = preprocessed_wav[self.attention_key]
            #return preprocessed_wav[self.features_key], preprocessed_wav[self.attention_key].bool()
        else:
            new_sample['attn_mask'] = None
        
        return new_sample