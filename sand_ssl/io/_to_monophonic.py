"""
Monochannel conversion

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch

class ToMonophonic(object):
    '''
    Convert to monochannel with a reduce function (can alter based on how waveform is loaded)
    :param reduce_fn: function to use for reducing channels (default = channel sum)
    '''
    def __init__(self, reduce_fn=None):
        if reduce_fn is None:
            self.reduce_fn = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
        else:
            self.reduce_fn = reduce_fn
        
    def __call__(self, sample:dict) -> dict:
        """
        Run monochannel conversion
        :param sample: dict, input sample
        :return monosample: dict, sample post monochannel conversion
        """
        monosample = sample.copy()
        
        waveform = monosample['waveform']
        #print(waveform.shape)
        waveform_mono = self.reduce_fn(waveform)
        #print(waveform_mono)
        #print(waveform_mono.shape)
        
        if waveform_mono.shape != torch.Size([1, waveform.shape[1]]):
            raise ValueError(f'Result of reduce_fn wrong shape, expected [1, {waveform.shape[1]}], got [{waveform_mono.shape[0], waveform_mono.shape[1]}]')
            
        monosample['waveform'] = waveform_mono
            
        return monosample