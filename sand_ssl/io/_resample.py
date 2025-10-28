"""
Resample transform

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torchaudio

class ResampleAudio(object):
    '''
    Resample a waveform
    :param resample_rate:int, rate to resample to (default=16000)
    '''
    def __init__(self, resample_rate: int = 16000):
        
        self.resample_rate = resample_rate
        
    def __call__(self, sample: dict) -> dict:
        """
        Run resampling
        :param sample: dict, sample
        :return resampled: dict, sample after resampling
        """
        resampled = sample.copy()    
        waveform, sample_rate = resampled['waveform'], resampled['sample_rate']
        if sample_rate != self.resample_rate:
            transform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)
            new_waveform = transform(waveform)
            resampled['waveform'] = new_waveform
            resampled['sample_rate'] = self.resample_rate
        
        return resampled