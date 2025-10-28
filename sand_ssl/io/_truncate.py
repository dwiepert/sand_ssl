"""
Truncate audio

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch 

class Truncate(object):
    '''
    Cut audio to specified length with optional offset
    :param length: float, length to trim to in terms of s
    :param offset: float, offset for clipping in terms of s (default = 0)
    :param pad: boolean, indicate whether to pad with truncation
    :param pad_method: str, indicate whether to pad with mean or pad with 0
    '''
    def __init__(self, length:float, offset:float = 0, pad:bool=False, pad_method:str='mean'):
        
        self.length = length
        self.offset = offset
        self.pad = pad
        self.pad_method = pad_method
        assert self.pad_method in ['mean', 'zero']
        
    def __call__(self, sample:dict) -> dict:
        """
        Truncate audio sample
        :param sample: dict, input sample
        :return clipsample: dict, sample after truncations
        """
        clipsample = sample.copy()
        waveform = clipsample['waveform']
        sr = clipsample['sample_rate']
        frames = int(self.length*sr)
        offset_frames = int(self.offset*sr)

        waveform_offset = waveform[:, offset_frames:]
        n_samples_remaining = waveform_offset.shape[1]
        
        if n_samples_remaining >= frames:
            waveform_trunc = waveform_offset[:, :frames]
        elif self.pad:
            n_channels = waveform_offset.shape[0]
            n_pad = frames - n_samples_remaining
            if self.pad_method == 'mean':
                channel_means = waveform_offset.mean(axis = 1).unsqueeze(1)
                add = torch.ones([n_channels, n_pad])*channel_means
            else:
                add = torch.zeros([n_channels, n_pad])
            waveform_trunc = torch.cat([waveform_offset, add], dim = 1)
        else:
            waveform_trunc = waveform_offset    
        clipsample['waveform'] = waveform_trunc
        
        return clipsample