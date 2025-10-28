"""
Custom collate function

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##built-in
from typing import List, Union
##third-party
import torch
##local
from sand_ssl.models import Extractor

def collate_features(batch: List[dict], feature_extractor:Union[Extractor]=None) -> dict:
    """
    Collate function for use when model has a feature extractor that automatically extracts to same length features

    :param batch: input batch, list of samples (dictionaries)
    :return sample: output sample dict, samples collated appropriately
    """
    uid = [item['uid'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    target_onehot = torch.stack([item['target_onehot'] for item in batch])
    target = torch.tensor([item['target'] for item in batch])
    if feature_extractor:
        wav_sample = {'waveform':[item['waveform'] for item in batch]}
        new_sample = feature_extractor(wav_sample)# torch.stack([item['waveform'] for item in batch])
        waveform = new_sample['waveform']
        attn_mask = new_sample['attn_mask']
    else:
        waveform = [item['waveform'] for item in batch]
        attn_mask = None
    
    sample = {'uid':uid, 'target_onehot':target_onehot, 'target':target, 'sample_rate':sr, 'waveform':waveform, 'attn_mask':attn_mask}

    return sample

def collate_wrapper(feature_extractor:Union[Extractor]):
    """
    Wrapper function for creating custom collate function with feature extractor included

    :param feature_extractor: initialized feature extractor
    :return: collate function
    """
    return lambda x: collate_features(x, feature_extractor=feature_extractor)
    